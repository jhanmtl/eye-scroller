import React from 'react';
import * as tf from "@tensorflow/tfjs"
import './App.css';
import {priors} from './priors'

class App extends React.Component {

    constructor(props) {
        super(props);

        this.priorsTensor=undefined;
        this.detectorModel=undefined;
        this.landmarksModel=undefined;

        this.cxhat=undefined;
        this.cyhat=undefined;
        this.what =undefined;
        this.hhat =undefined;

        this.confidence=undefined;

        this.gcx=undefined;
        this.gcy=undefined;
        this.gw =undefined;
        this.gh =undefined;

        this.bboxes=undefined;

        this.videoRef=React.createRef();

        this.inputCanvasRef=React.createRef();
        this.outputCanvasRef=React.createRef();
        this.leftCanvasRef=React.createRef();
        this.rightCanvasRef=React.createRef();

        this.inputCtx=undefined;
        this.outputCtx=undefined;
        this.leftCtx=undefined;
        this.rightCtx=undefined;

        this.leftTextRef=React.createRef();
        this.rightTextRef=React.createRef();
        this.fpsTextRef=React.createRef();

        this.videoHeight=undefined;
        this.videoWidth=undefined;
        this.videoOriginX=undefined;
        this.videoOriginY=undefined;
        this.cropDim=undefined;
        this.destDim=224;
        this.sf=0.75;

        this.threshold=0.55;
        this.nmsSigma=0.025;
    }

    componentDidMount() {
        this.priorsTensor=tf.tensor2d(priors);
        const priorSplits=tf.split(this.priorsTensor,4,1);

        this.cxhat=priorSplits[0]
        this.cyhat=priorSplits[1]
        this.what =priorSplits[2]
        this.hhat =priorSplits[3]

        tf.loadLayersModel("https://raw.githubusercontent.com/jhanmtl/blinker-fliper/master/public/detectorModel.json").then(loadedModel=>{
            this.detectorModel=loadedModel;
        });

        tf.loadLayersModel("https://raw.githubusercontent.com/jhanmtl/blinker-fliper/master/public/08-26-mobilenent-035-96x96.json").then(loadedModel=>{
            this.landmarksModel=loadedModel;
        });

        this.inputCtx=this.inputCanvasRef.current.getContext('2d');
        this.outputCtx=this.outputCanvasRef.current.getContext('2d');
        this.leftCtx=this.leftCanvasRef.current.getContext('2d');
        this.rightCtx=this.rightCanvasRef.current.getContext('2d');

        console.log(tf.getBackend())
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

                this.synchroPredict();
            })
        }
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
        this.confidence=confidence.reshape([36])

        let offsets=predictions[1]
        offsets=offsets.reshape([36,4])

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

    drawCenter(box,style){
        const cenY=(box[0]+box[2])/2;
        const cenX=(box[1]+box[3])/2;

        this.outputCtx.beginPath();
        this.outputCtx.arc(cenX,cenY,5,0,2*Math.PI,false);

        this.outputCtx.lineWidth=2;
        this.outputCtx.strokeStyle=style;
        this.outputCtx.stroke();
    }

    visualize(scoreA,scoreB,boxA,boxB){
        this.drawBox(boxA, "rgb(0,255,0)")
        this.drawBox(boxB, "rgb(0,255,0)")
        this.leftTextRef.current.innerText="left eye confidence: "+(100*scoreA).toFixed(2)+"%"
        this.rightTextRef.current.innerText="right eye confidence: "+(100*scoreB).toFixed(2)+"%"

        // this.drawCenter(boxA,"rgb(0,255,255)");
        // this.drawCenter(boxB,"rgb(0,255,255)");
    }

    updateOutputCanvas(){
        this.outputCtx.drawImage(this.inputCanvasRef.current,
                                0,
                                0,
                                this.destDim,
                                this.destDim,
                                0,
                                0,
                                this.destDim,
                                this.destDim);
    }

    prepDetectorInput(canvas){
        let imgTensor = tf.expandDims(tf.browser.fromPixels(canvas), 0);
        imgTensor = tf.cast(imgTensor, 'float32');
        imgTensor = tf.div(imgTensor, 127.5);
        imgTensor = tf.sub(imgTensor, 1);

        return imgTensor;
    }

    prepLandmarkInput(){
        let imgTensorLeft = tf.expandDims(tf.browser.fromPixels(this.leftCanvasRef.current), 0);
        imgTensorLeft = tf.cast(imgTensorLeft, 'float32');
        imgTensorLeft = tf.div(imgTensorLeft, 127.5);
        imgTensorLeft = tf.sub(imgTensorLeft, 1);

        let imgTensorRight = tf.expandDims(tf.browser.fromPixels(this.rightCanvasRef.current), 0);
        imgTensorRight = tf.cast(imgTensorRight, 'float32');
        imgTensorRight = tf.div(imgTensorRight, 127.5);
        imgTensorRight = tf.sub(imgTensorRight, 1);

        return tf.concat([imgTensorLeft,imgTensorRight],0);
    }

    getBoxesAndScores(predictions){
        this.parsePredictions(predictions);
        this.recoverBboxes();

        let boxes = this.bboxes.arraySync()
        let nmsResult = tf.image.nonMaxSuppressionWithScore(this.bboxes,
                                                            this.confidence,
                                                            2,
                                                            undefined,
                                                            undefined,
                                                            this.nmsSigma);
        let nmsIdx=nmsResult.selectedIndices;
        nmsIdx=nmsIdx.dataSync();
        let nmsScores=nmsResult.selectedScores;
        nmsScores=nmsScores.dataSync();

        let scoreA=nmsScores[0];
        let scoreB=nmsScores[1];

        let boxA;
        let boxB;

        if (boxes[nmsIdx[0]][1]<boxes[nmsIdx[1]][1]){
            boxA=boxes[nmsIdx[1]];
            boxB=boxes[nmsIdx[0]];
        }
        else{
            boxA=boxes[nmsIdx[0]];
            boxB=boxes[nmsIdx[1]];
        }

        return [scoreA,scoreB,boxA,boxB]
    }

    logFps(t0,t1){
        const elapsed=(t1-t0)/1000.0
        const fps=Math.round(1.0/elapsed)
        this.fpsTextRef.current.innerText="fps: "+fps
    }

    boxCrop=(box,ctx)=>{
        const cenX=(box[1]+box[3])/2
        const cenY=(box[0]+box[2])/2
        const h=box[2]-box[0]
        const w=box[3]-box[1]

        let cropSize=Math.max(h,w);
        cropSize+=10;

        const originX=cenX-cropSize/2;
        const originY=cenY-cropSize/2;

        ctx.drawImage(this.inputCanvasRef.current,
                    originX,
                    originY,
                    cropSize,
                    cropSize,
                    0,
                    0,
                    96,
                    96);
    }

    predictLandmarks(){
        const landmarksModelInput=this.prepLandmarkInput();
        let landmarkPredictions=this.landmarksModel.predict(landmarksModelInput);
        landmarkPredictions=landmarkPredictions.reshape([2,4,2]);
        landmarkPredictions=tf.cast(landmarkPredictions,'int32');
        return landmarkPredictions
    }

    drawLandmarks(landmarks,ctx,style){
        for (let i=0;i<landmarks.length;i++){
            ctx.beginPath();
            const landmark=landmarks[i];
            const x=landmark[0]
            const y=landmark[1]
            ctx.arc(x,y,2,0,2*Math.PI)
            ctx.fillStyle=style
            ctx.fill();
        }
    }

    synchroPredict=()=> {
        const t0 = performance.now()

        tf.tidy(() => {
                this.cropToCanvas();
                this.updateOutputCanvas();

                const detectorModelInput=this.prepDetectorInput(this.inputCanvasRef.current);
                const predictions = this.detectorModel.predict(detectorModelInput);
                const [scoreLeft,scoreRight,boxLeft,boxRight]=this.getBoxesAndScores(predictions);

                if (scoreLeft > this.threshold && scoreRight > this.threshold) {

                    this.visualize(scoreLeft,scoreRight,boxLeft,boxRight);

                    this.boxCrop(boxLeft,this.leftCtx);
                    this.boxCrop(boxRight,this.rightCtx);

                    const landmarkModelInput=this.prepLandmarkInput();
                    let landmarkPredictions=this.landmarksModel.predict(landmarkModelInput);
                    landmarkPredictions=tf.cast(landmarkPredictions,'int32');

                    let leftLandmarks=tf.gather(landmarkPredictions,0,0);
                    leftLandmarks=tf.reshape(leftLandmarks,[4,2])
                    leftLandmarks=leftLandmarks.arraySync();

                    let rightLandmarks=tf.gather(landmarkPredictions,1,0);
                    rightLandmarks=tf.reshape(rightLandmarks,[4,2])
                    rightLandmarks=rightLandmarks.arraySync();


                    this.drawLandmarks(leftLandmarks,this.leftCtx,'rgb(0,255,255)');
                    this.drawLandmarks(rightLandmarks,this.rightCtx,'rgb(0,255,255)');

                }
            })

        const t1=performance.now()
        this.logFps(t0,t1);
        window.requestAnimationFrame(this.synchroPredict)
    }


    render(){
        return (
            <div className="App">
                <p className="title">Blink Detector: tensorflowjs + react</p>

                <div className="container">
                    <div className="videoUnderlay">
                        <video id="webCam" autoPlay ref={this.videoRef}/>
                    </div>

                    <div className="input">
                        <canvas id='inputCanvasId' className="inputCanvas" width={224} height={224} ref={this.inputCanvasRef}/>
                    </div>


                <div className="output">
                        <canvas className="leftEyeCanvas" width={96} height={96} ref={this.leftCanvasRef}/>
                        <canvas className="outputCanvas" width={224} height={224} ref={this.outputCanvasRef}/>
                        <canvas className="rightEyeCanvas" width={96} height={96} ref={this.rightCanvasRef}/>

                </div>

                </div>


                <button onClick={this.startWebcam}>enable webcam</button>
                <p className="note" ref={this.leftTextRef}>left eye confidence:</p>
                <p className="note" ref={this.rightTextRef}>right eye confidence:</p>
                <p className="note" ref={this.fpsTextRef}>fps:</p>

                <a href="https://github.com/jhanmtl/eye-detector" >github repo</a>
            </div>
        );
    }

}

export default App;
