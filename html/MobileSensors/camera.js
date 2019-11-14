let pixelGrayScaleStd = 0;

/* This function checks and sets up the camera */
/* AmbientLightSensor */
function startVideo() {
  if (navigator.mediaDevices &&
    navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(handleUserMediaSuccess)
      .catch(handleUserMediaError);
  }

  /*determine light level: AmbientLightSensor.illuminance*/
  /*If you open a new browser tab on your phone and head over to chrome://flags/, scroll till you come across “Generic Sensor Extra Classes” and enable this flag*/
  const details = document.getElementById("details");

  let seenMaximum = 1;

  if (window.AmbientLightSensor) {
    try {
      const sensor = new AmbientLightSensor();

      console.log(`sensor: ${sensor}.`);

      // Detect changes in the light
      sensor.onreading = () => {
        //details.innerHTML = sensor.illuminance;
        seenMaximum = Math.max(seenMaximum, sensor.illuminance);

        console.log(`sensor.illuminance try`);
      
        // console.log(`sensor.illuminance: ${ Math.floor(toreading / seenMaximum * 10000) / 100 }`);

        $('#lightValue').html(Math.floor(sensor.illuminance / seenMaximum * 10000) / 100);      
        // $('#lightValue').html(sensor.illuminance);
        
        $('#cameraStatus').html(pixelGrayScaleStd < 5000 ? 'Yes' : 'No' );
      }

      // Has an error occured?
      sensor.onerror = event => document.getElementById("details").innerHTML = event.error.message;
      sensor.start();
    } catch (err) {
      details.innerHTML = err.message;
    }
  } else {
    details.innerHTML = 'It looks like your browser doesnt support this feature';
  }

}

function handleUserMediaError(error) {
  console.log(error);
}

/* This function initiates the camera video */
function handleUserMediaSuccess(stream) {

  var video = document.getElementById("myVideo");
  video.srcObject = stream;
  // video.play();

  /* We will capture an image twice every second */
  window.setInterval(captureImageFromVideo, 100);

  console.log("success");
}

/* This function stops the camera */
function endVideo() {
  myVideo.pause();
  console.log("Video off successfully");
}

/* This function captures the video */
function captureImageFromVideo() {
  var canvas = document.getElementById("mainCanvas");
  var context = canvas.getContext("2d");

  var video = document.getElementById("myVideo");
  canvas.setAttribute("width", video.width);
  canvas.setAttribute("height", video.height);
  // Draw video image onto Canvas
  context.drawImage(video, 0, 0, video.width, video.height);

  // Here is how you would process the image to turn it into a grayscale image  
  var dataObj = context.getImageData(0, 0, canvas.width, canvas.height);

  // Now data is a long, flat array containing the RGB values of each pixel
  var data = dataObj.data;

  //   console.log(`video data view: ${data}`)

  let sumGrayScale = 0;
  let grayScaleData = [];

  for (var i = 0; i < data.length; i += 4) {
    var gray = 0.333 * (data[i] + data[i + 1] + data[i + 2]);

    // Notice how this is not a 2D array the way we are
    // used to representing and indexing images (i, j).
    // Instead, it is flat with RGB cycling first, then
    // columns, then rows.
    data[i] = gray;
    data[i + 1] = gray;
    data[i + 2] = gray;

    sumGrayScale += gray;
    grayScaleData.push(gray);
  }

  let averagePixelGrayScale = sumGrayScale / grayScaleData.length ;

  // Standard deviation

  let sum = 0
  for (let v of grayScaleData) {
    const d = v - averagePixelGrayScale;
    sum += d * d;
  }

  pixelGrayScaleStd = Math.sqrt(sum);

  context.putImageData(dataObj, 0, 0);
}
