import React from 'react';
import * as tf from "@tensorflow/tfjs"
import './App.css';
import {priors} from './priors'
import Paper from '@material-ui/core/Paper';
import {MyButton} from "./components/MyButton";
import {SmallButton} from "./components/SmallButton";
import Scrollbar from "react-scrollbars-custom";
import {text} from "./text"

import 'chartjs-plugin-streaming'

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
        this.scrollRef=React.createRef();

        this.videoHeight=undefined;
        this.videoWidth=undefined;
        this.videoOriginX=undefined;
        this.videoOriginY=undefined;
        this.cropDim=undefined;
        this.destDim=224;
        this.sf=1.0;

        this.eyeDim=112;
        this.landmarkCount=4;

        this.detectThreshold=0.7;
        this.blinkThreshold=0.3;
        this.nmsSigma=0.025;

        this.leftRatioHistory=[];
        this.rightRatioHistory=[];

        this.state={
            scrollPos:50,
            inProgress:false,
        }
    }

    componentDidMount() {

        this.priorsTensor=tf.tensor2d(priors);
        const priorSplits=tf.split(this.priorsTensor,4,1);

        this.cxhat=priorSplits[0]
        this.cyhat=priorSplits[1]
        this.what =priorSplits[2]
        this.hhat =priorSplits[3]

        tf.loadLayersModel("https://raw.githubusercontent.com/jhanmtl/eye-scroller/master/public/detectorModel.json").then(loadedModel=>{
                                this.detectorModel=loadedModel;
                            });
        tf.loadLayersModel("https://raw.githubusercontent.com/jhanmtl/eye-scroller/master/public/landmarkModel.json").then(loadedModel=>{
            this.landmarksModel=loadedModel;
                            });

        this.inputCtx=this.inputCanvasRef.current.getContext('2d');
        this.outputCtx=this.outputCanvasRef.current.getContext('2d');
        this.leftCtx=this.leftCanvasRef.current.getContext('2d');
        this.rightCtx=this.rightCanvasRef.current.getContext('2d');

        this.fillBlack();
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
        this.setState(
            {
                inProgress:true
            }
        )
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
        this.outputCtx.lineWidth=1
        this.outputCtx.strokeRect(x,y,w,h)
    }

    visualize(scoreA,scoreB,boxA,boxB){
        this.drawBox(boxA, "rgb(0,255,255)")
        this.drawBox(boxB, "rgb(0,255,255)")
        this.leftTextRef.current.innerText="left eye confidence: "+(100*scoreA).toFixed(2)+"%"
        this.rightTextRef.current.innerText="right eye confidence: "+(100*scoreB).toFixed(2)+"%"
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

    prepLandmarkInputSingle(canvas){
        let imgTensor = tf.expandDims(tf.browser.fromPixels(canvas), 0);
        imgTensor = tf.cast(imgTensor, 'float32');
        imgTensor = tf.div(imgTensor, 127.5);
        imgTensor = tf.sub(imgTensor, 1);

        return imgTensor
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
                    this.eyeDim,
                    this.eyeDim);
    }

    predictLandmarks(inputCanvas){
        const input=this.prepLandmarkInputSingle(inputCanvas);
        let prediction=this.landmarksModel.predict(input);
        prediction=tf.cast(prediction,'int32');
        prediction=tf.reshape(prediction,[this.landmarkCount,2])
        let landmarks=prediction.arraySync();
        return landmarks;
    }

    fillBlack(){
        const blackBg='rgb(0,0,0)';
        this.leftCtx.fillStyle=blackBg;
        this.rightCtx.fillStyle=blackBg;
        this.leftCtx.fillRect(0,0,this.eyeDim,this.eyeDim);
        this.rightCtx.fillRect(0,0,this.eyeDim,this.eyeDim);
    }

    drawLandmarks(landmarks,ctx,style){
        ctx.fillStyle='rgb(0,0,0)'
        ctx.fillRect(0,0,this.eyeDim,this.eyeDim)

        for (let i=0;i<landmarks.length;i++){
            ctx.beginPath();
            const landmark=landmarks[i];
            const x=landmark[0]
            const y=landmark[1]
            ctx.arc(x,y,2,0,2*Math.PI)
            ctx.fillStyle=style
            ctx.fill();
        }

        for (let i=0;i<landmarks.length;i++){

            let startPt=landmarks[i];
            let endPt;

            if (i===landmarks.length-1){
                endPt=landmarks[0];
            }
            else{
                endPt=landmarks[i+1]
            }

            ctx.beginPath();
            ctx.moveTo(startPt[0],startPt[1])
            ctx.lineTo(endPt[0],endPt[1])
            ctx.strokeStyle=style
            ctx.stroke();
        }

        const topMarker=landmarks[1]
        const bottomMarker=landmarks[3]
        const centerX=(topMarker[0]+bottomMarker[0])/2
        const centerY=(topMarker[1]+bottomMarker[1])/2
        const height=Math.abs(bottomMarker[1]-topMarker[1])
        const radius=height/2

        const leftMarker=landmarks[0]
        const rightMarker=landmarks[2]
        const width=Math.abs(leftMarker[0]-rightMarker[0])

        ctx.beginPath();
        ctx.arc(centerX,centerY,radius,0,2*Math.PI);
        ctx.strokeStyle=style;
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(centerX,centerY,radius/2,0,2*Math.PI);
        ctx.fillStyle=style;
        ctx.fill();

        return height/(width+1e-3);
    }

    logRatioHistory=(nowLeftRatio, nowRightRatio)=>{
        if (this.leftRatioHistory.length<2){
            this.leftRatioHistory.push(nowLeftRatio);
        }
        else{
            this.leftRatioHistory[0]=this.leftRatioHistory[1];
            this.leftRatioHistory[1]=nowLeftRatio;
        }

        if (this.rightRatioHistory.length<2){
            this.rightRatioHistory.push(nowRightRatio);
        }
        else{
            this.rightRatioHistory[0]=this.rightRatioHistory[1];
            this.rightRatioHistory[1]=nowRightRatio;
        }
    }

    percentDiff=(ratioHistory)=>{
        let top=ratioHistory[0]-ratioHistory[1]
        let bottom=(ratioHistory[0]+1e-5)
        return top/bottom
    }

    synchroPredict=()=> {
        const t0 = performance.now()
        tf.tidy(() => {
                this.cropToCanvas();
                this.updateOutputCanvas();

                const detectorModelInput=this.prepDetectorInput(this.inputCanvasRef.current);
                const predictions = this.detectorModel.predict(detectorModelInput);
                const [scoreLeft,scoreRight,boxLeft,boxRight]=this.getBoxesAndScores(predictions);

                if (scoreLeft > this.detectThreshold && scoreRight > this.detectThreshold) {

                    this.visualize(scoreLeft,scoreRight,boxLeft,boxRight);

                    this.boxCrop(boxLeft,this.leftCtx);
                    this.boxCrop(boxRight,this.rightCtx);

                    let leftLandmarks=this.predictLandmarks(this.leftCanvasRef.current)
                    let currentLeftRatio=this.drawLandmarks(leftLandmarks,this.leftCtx,'rgb(0,255,255)');

                    let rightLandmarks=this.predictLandmarks(this.rightCanvasRef.current)
                    let currentRightRatio=this.drawLandmarks(rightLandmarks,this.rightCtx,'rgb(0,255,255)');

                    this.logRatioHistory(currentLeftRatio,currentRightRatio);

                    let leftChange=this.percentDiff(this.leftRatioHistory)
                    let rightChange=this.percentDiff(this.rightRatioHistory)

                    if (leftChange>this.blinkThreshold || rightChange>this.blinkThreshold){
                        this.scrollDown();
                    }
                }

                else{
                    this.fillBlack();
                }
            })

        const t1=performance.now()
        this.logFps(t0,t1);
        window.requestAnimationFrame(this.synchroPredict)
    }


    scrollDown=()=>{
        this.scrollRef.current.scrollTo(0, this.state.scrollPos);
        let newPos=Math.min(this.state.scrollPos+50, 4100)
        if (newPos<4100) {
            this.setState({
                scrollPos: newPos
            })
        }
        else{
             this.scrollRef.current.scrollToTop();
             this.setState({
                scrollPos: 50
            })
        }
    }

    scrollToTop=()=>{
       this.scrollRef.current.scrollToTop();
       this.setState({
           scrollPos:50
       })
    }

    formatText(){
        let paragraphs=[]
        for (let i=0;i<text.length;i++){
            if (i<5){
                paragraphs.push(
                    <p key={i} className="headerNotes">{text[i]}</p>
                )
            }
            else if (i===6){
                paragraphs.push(
                    <p key={i} className="smallNote">{text[i]}</p>
                )
            }
            else{
                 paragraphs.push(
                    <p key={i} className="note">{text[i]}</p>
                )
            }

        }
        return paragraphs
    }

    render(){

        return (
            <div className="App">
                <div className="overall">
                    <div className="controlPanel">
                        <Paper className="controlPaper" elevation={4}>
                            <p className="title">smart scroller: tensorflowjs + react</p>

                            <p className="note">eye detection</p>

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

                                <div className="output">
                                    {(this.state.inProgress)?<p className="loadingText">initiating model . . .</p>:""}
                                </div>

                                    <p className="note">eye extraction</p>

                                <div>
                                    <canvas className="leftEyeCanvas" width={this.eyeDim} height={this.eyeDim} ref={this.leftCanvasRef}/>
                                    <canvas className="rightEyeCanvas" width={this.eyeDim} height={this.eyeDim} ref={this.rightCanvasRef}/>
                                </div>

                                <MyButton onClick={this.startWebcam}>start model</MyButton>

                                <p className="note" ref={this.leftTextRef}>left eye confidence:</p>
                                <p className="note" ref={this.rightTextRef}>right eye confidence:</p>
                                <p className="note" ref={this.fpsTextRef}>fps:</p>

                                <a href="https://github.com/jhanmtl/eye-scroller" className="note">github repo</a>

                            </div>

                        </Paper>
                    </div>
                    <div className="textWindow">
                        <Paper className="textPaper" elevation={4}>
                            <div className="textContent">
                                <Scrollbar ref={this.scrollRef} minimalThumbSize={10} maximalThumbSize={12}>
                                    <div className="textBox">
                                        {this.formatText()}
                                    </div>
                                </Scrollbar>
                            </div>

                            <SmallButton onClick={this.scrollToTop} style={{}}>back to top</SmallButton>

                        </Paper>

                    </div>
                </div>


            </div>


        );
    }

}

export default App;
