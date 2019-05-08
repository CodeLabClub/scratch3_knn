const Runtime = require('../../engine/runtime');

const ArgumentType = require('../../extension-support/argument-type');
const BlockType = require('../../extension-support/block-type');
const Clone = require('../../util/clone');
const Cast = require('../../util/cast');
const Video = require('../../io/video');
const formatMessage = require('format-message');
import * as tf from '@tensorflow/tfjs';
import * as mobilenetModule from './mobilenet.js';
import * as knnClassifier from '@tensorflow-models/knn-classifier';

/**
 * Sensor attribute video sensor block should report.
 * @readonly
 * @enum {string}
 */
const SensingAttribute = {
    /** The amount of motion. */
    MOTION: 'motion',

    /** The direction of the motion. */
    DIRECTION: 'direction'
};

/**
 * Subject video sensor block should report for.
 * @readonly
 * @enum {string}
 */
const SensingSubject = {
    /** The sensor traits of the whole stage. */
    STAGE: 'Stage',

    /** The senosr traits of the area overlapped by this sprite. */
    SPRITE: 'this sprite'
};

/**
 * States the video sensing activity can be set to.
 * @readonly
 * @enum {string}
 */
const VideoState = {
    /** Video turned off. */
    OFF: 'off',

    /** Video turned on with default y axis mirroring. */
    ON: 'on',

    /** Video turned on without default y axis mirroring. */
    ON_FLIPPED: 'on-flipped'
};

/**
 * Class for the motion-related blocks in Scratch 3.0
 * @param {Runtime} runtime - the runtime instantiating this block package.
 * @constructor
 */
class Scratch3Knn {
    constructor(runtime) {
        this.knn = null
        this.trainTypes = ['A', 'B', 'C', 'D', 'E', 'F']

        this.knnInit()
        /**
         * The runtime instantiating this block package.
         * @type {Runtime}
         */
        this.runtime = runtime;

        /**
         * The last millisecond epoch timestamp that the video stream was
         * analyzed.
         * @type {number}
         */
        this._lastUpdate = null;

        if (this.runtime.ioDevices) {
            // Clear target motion state values when the project starts.
            this.runtime.on(Runtime.PROJECT_RUN_START, this.reset.bind(this));

            // Kick off looping the analysis logic.
            // this._loop();

            // Configure the video device with values from a globally stored
            // location.
            this.setVideoTransparency({
                TRANSPARENCY: this.globalVideoTransparency
            });
            this.videoToggle({
                VIDEO_STATE: this.globalVideoState
            });
        }

        setInterval(async () => {
            if (this.globalVideoState === VideoState.ON) {
                await this.gotResult()
                console.log('knn result:', this.trainResult)
            }
        }, 800)
    }

    /**
     * After analyzing a frame the amount of milliseconds until another frame
     * is analyzed.
     * @type {number}
     */
    static get INTERVAL() {
        return 33;
    }

    /**
     * Dimensions the video stream is analyzed at after its rendered to the
     * sample canvas.
     * @type {Array.<number>}
     */
    static get DIMENSIONS() {
        return [480, 360];
    }

    /**
     * The key to load & store a target's motion-related state.
     * @type {string}
     */
    static get STATE_KEY() {
        return 'Scratch.videoSensing';
    }

    /**
     * The default motion-related state, to be used when a target has no existing motion state.
     * @type {MotionState}
     */
    static get DEFAULT_MOTION_STATE() {
        return {
            motionFrameNumber: 0,
            motionAmount: 0,
            motionDirection: 0
        };
    }

    /**
     * The transparency setting of the video preview stored in a value
     * accessible by any object connected to the virtual machine.
     * @type {number}
     */
    get globalVideoTransparency() {
        const stage = this.runtime.getTargetForStage();
        if (stage) {
            return stage.videoTransparency;
        }
        return 50;
    }

    set globalVideoTransparency(transparency) {
        const stage = this.runtime.getTargetForStage();
        if (stage) {
            stage.videoTransparency = transparency;
        }
        return transparency;
    }

    /**
     * The video state of the video preview stored in a value accessible by any
     * object connected to the virtual machine.
     * @type {number}
     */
    get globalVideoState() {
        const stage = this.runtime.getTargetForStage();
        if (stage) {
            return stage.videoState;
        }
        return VideoState.ON;
    }

    set globalVideoState(state) {
        const stage = this.runtime.getTargetForStage();
        if (stage) {
            stage.videoState = state;
        }
        return state;
    }

    /**
     * Reset the extension's data motion detection data. This will clear out
     * for example old frames, so the first analyzed frame will not be compared
     * against a frame from before reset was called.
     */
    reset() {
        const targets = this.runtime.targets;
        for (let i = 0; i < targets.length; i++) {
            const state = targets[i].getCustomState(Scratch3Knn .STATE_KEY);
            if (state) {
                state.motionAmount = 0;
                state.motionDirection = 0;
            }
        }
    }

    /**
     * Occasionally step a loop to sample the video, stamp it to the preview
     * skin, and add a TypedArray copy of the canvas's pixel data.
     * @private
     */
    _loop() {
        setTimeout(this._loop.bind(this), Math.max(this.runtime.currentStepTime, Scratch3Knn .INTERVAL));

        // Add frame to detector
        const time = Date.now();
        if (this._lastUpdate === null) {
            this._lastUpdate = time;
        }
        const offset = time - this._lastUpdate;
        if (offset > Scratch3Knn .INTERVAL) {
            const frame = this.runtime.ioDevices.video.getFrame({
                format: Video.FORMAT_IMAGE_DATA,
                dimensions: Scratch3Knn .DIMENSIONS
            });
            if (frame) {
                this._lastUpdate = time;
            }
        }
    }

    /**
     * Create data for a menu in scratch-blocks format, consisting of an array
     * of objects with text and value properties. The text is a translated
     * string, and the value is one-indexed.
     * @param {object[]} info - An array of info objects each having a name
     *   property.
     * @return {array} - An array of objects with text and value properties.
     * @private
     */
    _buildMenu(info) {
        return info.map((entry, index) => {
            const obj = {};
            obj.text = entry.name;
            obj.value = entry.value || String(index + 1);
            return obj;
        });
    }

    /**
     * @param {Target} target - collect motion state for this target.
     * @returns {MotionState} the mutable motion state associated with that
     *   target. This will be created if necessary.
     * @private
     */
    _getMotionState(target) {
        let motionState = target.getCustomState(Scratch3Knn .STATE_KEY);
        if (!motionState) {
            motionState = Clone.simple(Scratch3Knn .DEFAULT_MOTION_STATE);
            target.setCustomState(Scratch3Knn .STATE_KEY, motionState);
        }
        return motionState;
    }

    static get SensingAttribute() {
        return SensingAttribute;
    }

    /**
     * An array of choices of whether a reporter should return the frame's
     * motion amount or direction.
     * @type {object[]} an array of objects
     * @param {string} name - the translatable name to display in sensor
     *   attribute menu
     * @param {string} value - the serializable value of the attribute
     */
    get ATTRIBUTE_INFO() {
        return [
            {
                name: 'motion',
                value: SensingAttribute.MOTION
            },
            {
                name: 'direction',
                value: SensingAttribute.DIRECTION
            }
        ];
    }

    static get SensingSubject() {
        return SensingSubject;
    }

    /**
     * An array of info about the subject choices.
     * @type {object[]} an array of objects
     * @param {string} name - the translatable name to display in the subject menu
     * @param {string} value - the serializable value of the subject
     */
    get SUBJECT_INFO() {
        return [
            {
                name: 'stage',
                value: SensingSubject.STAGE
            },
            {
                name: 'sprite',
                value: SensingSubject.SPRITE
            }
        ];
    }

    /**
     * States the video sensing activity can be set to.
     * @readonly
     * @enum {string}
     */
    static get VideoState() {
        return VideoState;
    }

    /**
     * An array of info on video state options for the "turn video [STATE]" block.
     * @type {object[]} an array of objects
     * @param {string} name - the translatable name to display in the video state menu
     * @param {string} value - the serializable value stored in the block
     */
    get VIDEO_STATE_INFO() {
        return [
            {
                name: 'off',
                value: VideoState.OFF
            },
            {
                name: 'on',
                value: VideoState.ON
            },
            {
                name: 'on flipped',
                value: VideoState.ON_FLIPPED
            }
        ];
    }


    /**
     * @returns {object} metadata for this extension and its blocks.
     */
    getInfo() {
        return {
            id: 'cxknn',
            name: 'KNN Classifier',
            blocks: [
                {
                    opcode: 'videoToggle',
                    text: 'turn video [VIDEO_STATE]',
                    arguments: {
                        VIDEO_STATE: {
                            type: ArgumentType.NUMBER,
                            menu: 'VIDEO_STATE',
                            defaultValue: VideoState.ON
                        }
                    }
                },
                {
                    opcode: 'setVideoTransparency',
                    text: 'set video transparency to [TRANSPARENCY]',
                    arguments: {
                        TRANSPARENCY: {
                            type: ArgumentType.NUMBER,
                            defaultValue: 50
                        }
                    }
                },
                {
                    opcode: 'isloaded',
                    blockType: BlockType.REPORTER,
                    text: formatMessage({
                        id: 'knn.isloaded',
                        default: 'is loaded',
                        description: 'knn is loaded'
                    })
                },
                {
                    opcode: 'trainA',
                    blockType: BlockType.COMMAND,
                    text: formatMessage({
                        id: 'knn.trainA',
                        default: 'Train label 1 [STRING]',
                        description: 'Train A'
                    }),
                    arguments: {
                        STRING: {
                            type: ArgumentType.STRING,
                            defaultValue: "A"
                        }
                    }
                },
                {
                    opcode: 'trainB',
                    blockType: BlockType.COMMAND,
                    text: formatMessage({
                        id: 'knn.trainB',
                        default: 'Train label 2 [STRING]',
                        description: 'Train B'
                    }),
                    arguments: {
                        STRING: {
                            type: ArgumentType.STRING,
                            defaultValue: "B"
                        }
                    }
                },
                {
                    opcode: 'trainC',
                    blockType: BlockType.COMMAND,
                    text: formatMessage({
                        id: 'knn.trainC',
                        default: 'Train label 3 [STRING]',
                        description: 'Train C'
                    }),
                    arguments: {
                        STRING: {
                            type: ArgumentType.STRING,
                            defaultValue: "C"
                        }
                    }
                },
                {
                    opcode: 'trainD',
                    blockType: BlockType.COMMAND,
                    text: formatMessage({
                        id: 'knn.trainD',
                        default: 'Train label 4 [STRING]',
                        description: 'Train D'
                    }),
                    arguments: {
                        STRING: {
                            type: ArgumentType.STRING,
                            defaultValue: "D"
                        }
                    }
                },
                {
                    opcode: 'trainE',
                    blockType: BlockType.COMMAND,
                    text: formatMessage({
                        id: 'knn.trainE',
                        default: 'Train label 5 [STRING]',
                        description: 'Train E'
                    }),
                    arguments: {
                        STRING: {
                            type: ArgumentType.STRING,
                            defaultValue: "E"
                        }
                    }
                },
                {
                    opcode: 'trainF',
                    blockType: BlockType.COMMAND,
                    text: formatMessage({
                        id: 'knn.trainF',
                        default: 'Train label 6 [STRING]',
                        description: 'Train F'
                    }),
                    arguments: {
                        STRING: {
                            type: ArgumentType.STRING,
                            defaultValue: "F"
                        }
                    }
                },
                {
                    opcode: 'resetTrain',
                    blockType: BlockType.COMMAND,
                    text: formatMessage({
                        id: 'knn.reset',
                        default: 'Reset [STRING]',
                        description: 'reset'
                    }),
                    arguments: {
                        STRING: {
                            type: ArgumentType.STRING,
                            defaultValue: "A"
                        }
                    }
                },
                {
                    opcode: 'Samples',
                    blockType: BlockType.REPORTER,
                    text: formatMessage({
                        id: 'knn.samples',
                        default: 'Samples [STRING]',
                        description: 'samples'
                    }),
                    arguments: {
                        STRING: {
                            type: ArgumentType.STRING,
                            defaultValue: "A"
                        }
                    }
                },
                {
                    opcode: 'getResult',
                    blockType: BlockType.REPORTER,
                    text: formatMessage({
                        id: 'knn.getResult',
                        default: 'Result',
                        description: 'getResult'
                    }),
                    arguments: {

                    }
                },
                {
                    opcode: 'getConfidence',
                    blockType: BlockType.REPORTER,
                    text: formatMessage({
                        id: 'knn.getConfidence',
                        default: 'getConfidence [STRING]',
                        description: 'getConfidence'
                    }),
                    arguments: {
                        STRING: {
                            type: ArgumentType.STRING,
                            defaultValue: "A"
                        }
                    }
                },
                {
                    opcode: 'whenGetResult',
                    blockType: BlockType.HAT,
                    text: formatMessage({
                        id: 'knn.whenGetResult',
                        default: 'when get [STRING]',
                        description: 'whenGetResult'
                    }),
                    arguments: {
                        STRING: {
                            type: ArgumentType.STRING,
                            defaultValue: "A"
                        }
                    }
                }
            ],
            menus: {
                ATTRIBUTE: this._buildMenu(this.ATTRIBUTE_INFO),
                SUBJECT: this._buildMenu(this.SUBJECT_INFO),
                VIDEO_STATE: this._buildMenu(this.VIDEO_STATE_INFO)
            }
        };
    }


    /**
     * A scratch command block handle that configures the video state from
     * passed arguments.
     * @param {object} args - the block arguments
     * @param {VideoState} args.VIDEO_STATE - the video state to set the device to
     */
    videoToggle(args) {
        const state = args.VIDEO_STATE;
        this.globalVideoState = state;
        if (state === VideoState.OFF) {
            this.runtime.ioDevices.video.disableVideo();
        } else {
            this.runtime.ioDevices.video.enableVideo();
            // Mirror if state is ON. Do not mirror if state is ON_FLIPPED.
            this.runtime.ioDevices.video.mirror = state === VideoState.ON;
        }
    }

    /**
     * A scratch command block handle that configures the video preview's
     * transparency from passed arguments.
     * @param {object} args - the block arguments
     * @param {number} args.TRANSPARENCY - the transparency to set the video
     *   preview to
     */
    setVideoTransparency(args) {
        const transparency = Cast.toNumber(args.TRANSPARENCY);
        this.globalVideoTransparency = transparency;
        this.runtime.ioDevices.video.setPreviewGhost(transparency);
    }

    clearClass(classIndex) {
        this.classifier.clearClass(classIndex);
    }

    updateExampleCounts(args, util) {
        let counts = this.classifier.getClassExampleCount();
        this.runtime.emit('SAY', util.target, 'say', this.trainTypes[0] + '样本数：' + (counts[0] || 0) + '\n'
        + this.trainTypes[1] + '样本数：' + (counts[1] || 0) + '\n'
        + this.trainTypes[2] + '样本数：' + (counts[2] || 0) + '\n'
        + this.trainTypes[3] + '样本数：' + (counts[3] || 0) + '\n'
        + this.trainTypes[4] + '样本数：' + (counts[4] || 0) + '\n'
        + this.trainTypes[5] + '样本数：' + (counts[5] || 0));
        console.log('A样本数：', counts[0], 'B样本数：', counts[1], 'C样本数：', counts[2], 'D样本数：', counts[3], 'E样本数：', counts[4], 'F样本数：', counts[5])
        const _target = util.target;
    }

    isloaded() {
        return Boolean(this.mobilenet)
    }

    trainA(args, util) {
        if (this.globalVideoState === VideoState.OFF) {
            alert('请先打开摄像头')
            return
        }
        let img = document.createElement('img')
        img.src = this.runtime.ioDevices.video.getFrame({
            format: Video.FORMAT_CANVAS,
            dimensions: Scratch3Knn.DIMENSIONS
        }).toDataURL("image/png")
        img.width = 480
        img.height = 360
        img.onload = () => {
            const img0 = tf.fromPixels(img);
            const logits0 = this.mobilenet.infer(img0, 'conv_preds');
            this.classifier.addExample(logits0, 0);
            this.trainTypes[0] = args.STRING
            this.updateExampleCounts(args, util);
        }
    }

    trainB(args, util) {
        if (this.globalVideoState === VideoState.OFF) {
            alert('请先打开摄像头')
            return
        }
        let img = document.createElement('img')
        img.src = this.runtime.ioDevices.video.getFrame({
            format: Video.FORMAT_CANVAS,
            dimensions: Scratch3Knn.DIMENSIONS
        }).toDataURL("image/png")
        img.width = 480
        img.height = 360
        img.onload = () => {
            const img0 = tf.fromPixels(img);
            const logits0 = this.mobilenet.infer(img0, 'conv_preds');
            this.classifier.addExample(logits0, 1);
            this.trainTypes[1] = args.STRING
            this.updateExampleCounts(args, util);
        }
    }

    trainC(args, util) {
        if (this.globalVideoState === VideoState.OFF) {
            alert('请先打开摄像头')
            return
        }
        let img = document.createElement('img')
        img.src = this.runtime.ioDevices.video.getFrame({
            format: Video.FORMAT_CANVAS,
            dimensions: Scratch3Knn.DIMENSIONS
        }).toDataURL("image/png")
        img.width = 480
        img.height = 360
        img.onload = () => {
            const img0 = tf.fromPixels(img);
            const logits0 = this.mobilenet.infer(img0, 'conv_preds');
            this.classifier.addExample(logits0, 2);
            this.trainTypes[2] = args.STRING
            this.updateExampleCounts(args, util);
        }
    }

    trainD(args, util) {
        if (this.globalVideoState === VideoState.OFF) {
            alert('请先打开摄像头')
            return
        }
        let img = document.createElement('img')
        img.src = this.runtime.ioDevices.video.getFrame({
            format: Video.FORMAT_CANVAS,
            dimensions: Scratch3Knn.DIMENSIONS
        }).toDataURL("image/png")
        img.width = 480
        img.height = 360
        img.onload = () => {
            const img0 = tf.fromPixels(img);
            const logits0 = this.mobilenet.infer(img0, 'conv_preds');
            this.classifier.addExample(logits0, 3);
            this.trainTypes[3] = args.STRING
            this.updateExampleCounts(args, util);
        }
    }

    trainE(args, util) {
        if (this.globalVideoState === VideoState.OFF) {
            alert('请先打开摄像头')
            return
        }
        let img = document.createElement('img')
        img.src = this.runtime.ioDevices.video.getFrame({
            format: Video.FORMAT_CANVAS,
            dimensions: Scratch3Knn.DIMENSIONS
        }).toDataURL("image/png")
        img.width = 480
        img.height = 360
        img.onload = () => {
            const img0 = tf.fromPixels(img);
            const logits0 = this.mobilenet.infer(img0, 'conv_preds');
            this.classifier.addExample(logits0, 4);
            this.trainTypes[4] = args.STRING
            this.updateExampleCounts(args, util);
        }
    }

    trainF(args, util) {
        if (this.globalVideoState === VideoState.OFF) {
            alert('请先打开摄像头')
            return
        }
        let img = document.createElement('img')
        img.src = this.runtime.ioDevices.video.getFrame({
            format: Video.FORMAT_CANVAS,
            dimensions: Scratch3Knn.DIMENSIONS
        }).toDataURL("image/png")
        img.width = 480
        img.height = 360
        img.onload = () => {
            const img0 = tf.fromPixels(img);
            const logits0 = this.mobilenet.infer(img0, 'conv_preds');
            this.classifier.addExample(logits0, 5);
            this.trainTypes[5] = args.STRING
            this.updateExampleCounts(args, util);
        }
    }
    Samples(args, util) {
        let counts = this.classifier.getClassExampleCount();
        let index = this.trainTypes.indexOf(args.STRING)
        return counts[index] || 0
    }
    resetTrain(args, util) {
        let counts = this.classifier.getClassExampleCount();
        let index = this.trainTypes.indexOf(args.STRING)
        if (!counts[index]) {
            alert('该类别无训练数据')
            return
        }
        if (index < 0) {
            alert('未找到对应类别')
            return
        }
        this.clearClass(index);
        this.updateExampleCounts(args, util);
    }

    getResult(args, util) {
        return this.trainResult
    }
    getConfidence(args, util) {
        let index = this.trainTypes.indexOf(args.STRING)
        if (index === -1) {
            return 0
        }
        return (this.trainConfidences && this.trainConfidences[index]) || 0
    }
    gotResult(args, util) {
        return new Promise((resolve, reject) => {
            let img = document.createElement('img')
            let frame = this.runtime.ioDevices.video.getFrame({
                format: Video.FORMAT_CANVAS,
                dimensions: Scratch3Knn.DIMENSIONS
            })
            if (!Object.keys(this.classifier.getClassExampleCount()).length) {
                resolve()
                return
            }
            if (frame) {
                img.src = frame.toDataURL("image/png")
            } else {
                resolve()
                return
            }
            img.width = 480
            img.height = 360
            img.onload = async () => {
                const x = tf.fromPixels(img);
                const xlogits = this.mobilenet.infer(x, 'conv_preds');
                console.log('Predictions:');
                let res = await this.classifier.predictClass(xlogits);
                console.log(this.classifier.getClassExampleCount(), res)
                this.trainResult = this.trainTypes[res.classIndex] || 0
                this.trainConfidences = res.confidences
                resolve(this.trainResult)
            }
        })
    }

    whenGetResult(args, util) {
        if (this.trainResult === undefined) {
            return false
        }
        return args.STRING === this.trainResult
    }

    async knnInit () {
        this.classifier = knnClassifier.create();
        this.mobilenet = await mobilenetModule.load();
    }
}

module.exports = Scratch3Knn;
