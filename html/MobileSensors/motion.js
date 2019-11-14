
/*This function is called when the page is loaded*/
function handleLoadEvent() {
  if (window.DeviceMotionEvent) {
    window.addEventListener("devicemotion", handleDeviceMotionEvent);
    window.addEventListener("devicemotion", (event) => {
      const displayArea = $('#shakeDetection');
      if(detectShake(event)) {
        displayArea.html('Shake! Shake! Shake!');

        window.setTimeout(() => {
          displayArea.html('no shake detected');
        }, 3000);
      } 
    });
  }
  else {
    var messageDiv = document.getElementById("message");
    messageDiv.innerHTML = "DeviceMotionEvent not supported";
  }
}



/*This function handles deviceorientation events*/
function handleDeviceMotionEvent(event) {
  var messageDiv = document.getElementById("message");

  // Acceleration (meters per second squared)
  var accX = event.acceleration.x;
  var accY = event.acceleration.y;
  var accZ = event.acceleration.z;

  // Acceleration including gravity
  var accXwG = event.accelerationIncludingGravity.x;
  var accYwG = event.accelerationIncludingGravity.y;
  var accZwG = event.accelerationIncludingGravity.z;

  // Rotation rate (radians per second squared)
  var accDX = event.rotationRate.alpha;
  var accDY = event.rotationRate.beta;
  var accDZ = event.rotationRate.gamma;
  
  /* Displaying only the first set of values*/
  messageDiv.innerHTML = "accX:" + accX.toFixed(1) + " m/s^2 <br>";
  messageDiv.innerHTML += "accY:" + accY.toFixed(1) + " m/s^2 <br>";
  messageDiv.innerHTML += "accZ:" + accZ.toFixed(1) + " m/s^2";
}


let lastX, lastY, lastZ;
let moveDiscrepency = 0;

function detectShake(e) {
  let acc = e.acceleration;
  let shaked = false;

  // Acceleration (meters per second squared)
  var accX = event.acceleration.x;
  var accY = event.acceleration.y;
  var accZ = event.acceleration.z;

	if(Math.abs(accX) >= 1 &&
	Math.abs(accY) >= 1 &&
	Math.abs(accZ) >=1) {
		if(!lastX) {
			lastX = accX;
			lastY = accY;
			lastZ = accZ;
			return shaked;
		}

		let deltaX = Math.abs(accX - lastX);
		let deltaY = Math.abs(accY - lastY);
		let deltaZ = Math.abs(accZ - lastZ);
		
		if(deltaX + deltaY + deltaZ > 3) {
			moveDiscrepency++;
		} else {
			moveDiscrepency = Math.max(--moveDiscrepency, 0);
		}

		if(moveDiscrepency > 2) {
      shaked = true;
      moveDiscrepency = 0;			
		}

		lastX = accX;
		lastY = accY;
		lastZ = accZ;
    
    return shaked;
	}
}