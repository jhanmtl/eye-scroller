import React from 'react';
import * as tf from "@tensorflow/tfjs"
import './App.css';
import {priors} from './priors'

class App extends React.Component {

    constructor(props) {
        super(props);

        this.modelHasLoaded=false
        this.priorsTensor=undefined
        this.model=undefined

        this.cxhat=undefined
        this.cyhat=undefined
        this.what =undefined
        this.hhat =undefined

        this.confidence=undefined;

        this.gcx=undefined;
        this.gcy=undefined;
        this.gw =undefined;
        this.gh =undefined;

        this.bboxes=undefined;

        this.videoRef=React.createRef()
        this.inputCanvasRef=React.createRef()
        this.outputCanvasRef=React.createRef()
        this.leftTextRef=React.createRef()
        this.rightTextRef=React.createRef()

        this.inputCtx=undefined;
        this.outputCtx=undefined;

        this.videoHeight=undefined;
        this.videoWidth=undefined;
        this.videoOriginX=undefined;
        this.videoOriginY=undefined;
        this.cropDim=undefined;
        this.destDim=224
        this.sf=1

        this.threshold=0.5;
    }

    componentDidMount() {
        this.priorsTensor=tf.tensor2d(priors);
        const priorSplits=tf.split(this.priorsTensor,4,1);

        this.cxhat=priorSplits[0]
        this.cyhat=priorSplits[1]
        this.what =priorSplits[2]
        this.hhat =priorSplits[3]

        tf.loadLayersModel("https://raw.githubusercontent.com/jhanmtl/eye-detector/master/public/model.json").then(loadedModel=>{
            this.modelHasLoaded=true;
            this.model=loadedModel;
        });

        this.inputCtx=this.inputCanvasRef.current.getContext('2d');
        this.outputCtx=this.outputCanvasRef.current.getContext('2d');
    }

    cropToCanvas=()=>{
        this.inputCtx.drawImage(this.videoRef.current,
                                this.videoOriginX,
                                this.videoOriginY,
                                this.cropDim,
                                this.cropDim,
                                0,
                                0,
                                this.destDim,
                                this.destDim)
    }

    parsePredictions=(predictions)=>{
        let confidence=predictions[0]
        this.confidence=confidence.reshape([147])

        let offsets=predictions[1]
        offsets=offsets.reshape([147,4])

        const splits=tf.split(offsets,4,1)
        this.gcx=splits[0]
        this.gcy=splits[1]
        this.gw =splits[2]
        this.gh =splits[3]
    }

    recoverBboxes=()=>{
        const cx=tf.add(tf.mul(this.gcx,this.what),this.cxhat)
        const cy=tf.add(tf.mul(this.gcy,this.hhat),this.cyhat)
        const w= tf.mul(tf.exp(this.gw),this.what)
        const h= tf.mul(tf.exp(this.gh),this.hhat)

        const x1=tf.cast(tf.mul(tf.sub(cx,tf.div(w,2)),224),'int32')
        const y1=tf.cast(tf.mul(tf.sub(cy,tf.div(h,2)),224),'int32')
        const x2=tf.cast(tf.mul(tf.add(cx,tf.div(w,2)),224),'int32')
        const y2=tf.cast(tf.mul(tf.add(cy,tf.div(h,2)),224),'int32')

        this.bboxes=tf.concat([y1,x1,y2,x2],1)
    }

    drawBox(box,style){
        const x=box[1]
        const y=box[0]
        const w=box[3]-box[1]
        const h=box[2]-box[0]

        this.outputCtx.strokeStyle=style
        this.outputCtx.lineWidth=2
        this.outputCtx.strokeRect(x,y,w,h)
    }
    predict=()=>{

        tf.tidy(()=>{
                this.cropToCanvas();
                const imgTensor = tf.expandDims(tf.browser.fromPixels(this.inputCanvasRef.current))
                const predictions = this.model.predict(imgTensor);
                this.parsePredictions(predictions);
                this.recoverBboxes();

                let boxesPromise = this.bboxes.array()
                let nmsPromise = tf.image.nonMaxSuppressionWithScoreAsync(this.bboxes, this.confidence, 2, undefined, undefined, 0.25);

                Promise.all([boxesPromise, nmsPromise]).then(values => {

                    this.outputCtx.drawImage(this.videoRef.current,
                        this.videoOriginX,
                        this.videoOriginY,
                        this.cropDim,
                        this.cropDim,
                        0,
                        0,
                        this.destDim,
                        this.destDim)

                    let boxesVal = values[0]
                    let nmsIdxPromise = values[1].selectedIndices.array()
                    let nmsScorePromise = values[1].selectedScores.array()

                    Promise.all([nmsIdxPromise, nmsScorePromise]).then(values => {
                        let nmsIdx = values[0]
                        let nmsScores = values[1]

                        let boxA = boxesVal[nmsIdx[0]]
                        let scoreA = nmsScores[0]

                        let boxB = boxesVal[nmsIdx[1]]
                        let scoreB = nmsScores[1]

                        if (scoreA > this.threshold && scoreB > this.threshold) {
                            this.drawBox(boxA, "rgb(255,0,0)")
                            this.drawBox(boxB, "rgb(0,255,0)")
                            this.leftTextRef.current.innerText="left eye confidence: "+(100*scoreA).toFixed(2)+"%"
                            this.rightTextRef.current.innerText="right eye confidence: "+(100*scoreB).toFixed(2)+"%"
                        }
                    })
                })

                window.requestAnimationFrame(this.predict)
            })
    }

    setCropProps=(stream)=>{
        this.videoHeight=stream.getVideoTracks()[0].getSettings().height;
        this.videoWidth =stream.getVideoTracks()[0].getSettings().width;

        const cen_x=this.videoWidth/2
        const cen_y=this.videoHeight/2

        this.cropDim=this.destDim/this.sf
        this.videoOriginX=cen_x-this.cropDim/2
        this.videoOriginY=cen_y-this.cropDim/2
    }

    startWebcam=()=>{
        if (navigator.mediaDevices.getUserMedia){
            navigator.mediaDevices.getUserMedia({
                video:true,
                audio:false,
            }).then(stream=>{
                this.videoRef.current.srcObject=stream;
                this.videoRef.current.style.display='none'
                this.inputCanvasRef.current.style.display='none'
                this.setCropProps(stream)

                this.predict();
            })
        }
    }

    render(){
        return (
            <div className="App">
                <p className="title">Eye Detector: tensorflowjs + react</p>

                <div className="container">
                    <div className="videoUnderlay">
                        <video id="webCam" autoPlay ref={this.videoRef}/>
                    </div>

                    <div className="input">
                        <canvas id='inputCanvasId' className="inputCanvas" width={224} height={224} ref={this.inputCanvasRef}/>
                    </div>

                    <div className="output">
                        <canvas className="outputCanvas" width={224} height={224} ref={this.outputCanvasRef}/>
                    </div>

                </div>

                <button onClick={this.startWebcam}>enable webcam</button>
                <p className="note" ref={this.leftTextRef}>left eye confidence:</p>
                <p className="note" ref={this.rightTextRef}>right eye confidence:</p>
            </div>
        );
    }

}

export default App;
