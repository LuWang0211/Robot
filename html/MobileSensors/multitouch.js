

let touchX,touchY;
let lastX,lastY=-1;

// Set-up the canvas and add our event handlers after the page has loaded
function handleLoadEvent() {
    console.log(`start`)
    // Get the specific canvas element from the HTML document
    canvas = document.getElementById('draw_canvas');

    // If the browser supports the canvas tag, get the 2d drawing context for this canvas
    if (canvas.getContext){
        ctx = canvas.getContext('2d');
    }
    console.log(`get canvas, ctx: ${canvas}, ${ctx}`)
    // Check that we have a valid context to draw on/with before adding event handlers
    if (ctx) {
        canvas.addEventListener("touchstart", sketchpad_touchStart, false);
        canvas.addEventListener('touchend', sketchpad_touchEnd, false); // only use when drawLine
        canvas.addEventListener("touchmove", sketchpad_touchMove, false);
    }
}

function clearCanvas(canvas,ctx) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

// // drawDot
// function drawDot(ctx,x,y,size) {
//     // Let's use black by setting RGB values to 0, and 255 alpha (completely opaque)
//     r=0; g=0; b=0; a=255;

//     // Select a fill style
//     ctx.fillStyle = "rgba("+r+","+g+","+b+","+(a/255)+")";

//     // Draw a filled circle
//     ctx.beginPath();
//     ctx.arc(x, y, size, 0, Math.PI*2, true); 
//     ctx.closePath();
//     ctx.fill();
// } 

// drawLine
function drawLine(ctx,x,y,size) {

    // If lastX is not set, set lastX and lastY to the current position 
    if (lastX==-1) {
        lastX=x;
    lastY=y;
    }

    // Let's use black by setting RGB values to 0, and 255 alpha (completely opaque)
    r=0; g=0; b=0; a=255;

    // Select a fill style
    ctx.strokeStyle = "rgba("+r+","+g+","+b+","+(a/255)+")";

    // Set the line "cap" style to round, so lines at different angles can join into each other
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(lastX,lastY);

    ctx.lineTo(x,y);
        ctx.lineWidth = size;
        ctx.stroke();

        ctx.closePath();
    lastX=x;
    lastY=y;
}

function sketchpad_touchStart(event) {
    console.log(`sketchpad_touchStart`)
    getTouchPos();
    // drawDot(ctx,touchX,touchY,5);
    drawLine(ctx,touchX,touchY,5);
    console.log(`sketchpad_touchStart X, Y: ${touchX}, ${touchY}`)
    event.preventDefault();
}

// Reset lastX and lastY to -1, only use on Drawline, not drawdot
function sketchpad_touchEnd() {
    
    lastX=-1;
    lastY=-1;
}

function sketchpad_touchMove(e) { 
    console.log(`sketchpad_touchMove`)
    // Update the touch co-ordinates
    getTouchPos(e);

    // During a touchmove event, unlike a mousemove event, we don't need to check if the touch is engaged, since there will always be contact with the screen by definition.
    // drawDot(ctx,touchX,touchY,5);
    drawLine(ctx,touchX,touchY,5); 
    console.log(`sketchpad_touchMove X, Y: ${touchX}, ${touchY}`)

    // Prevent a scrolling action as a result of this touchmove triggering.
    event.preventDefault();
}

// Get the touch position relative to the top-left of the canvas
function getTouchPos(e) {
    if (!e)
        var e = event;

    if (e.touches) {
        if (e.touches.length == 1) { // Only deal with one finger
            let touch = e.touches[0]; // Get the information for finger #1
            touchX=touch.pageX-touch.target.offsetLeft;
            touchY=touch.pageY-touch.target.offsetTop;
        }
    }
}