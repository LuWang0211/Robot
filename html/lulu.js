/*The object that holds the img data inside JavaScript*/
var imgInfo = [
    "https://homes.cs.washington.edu/~mcakmak/faces/close_spacing_aido.png",
    "https://homes.cs.washington.edu/~mcakmak/faces/default_face_aido.png",
    "https://homes.cs.washington.edu/~mcakmak/faces/no_mouth_aido.png",
    "https://homes.cs.washington.edu/~mcakmak/faces/no_pupil_aido.png",
    "https://homes.cs.washington.edu/~mcakmak/faces/white_face_aido.png",
    "https://homes.cs.washington.edu/~mcakmak/faces/wide_spacing_aido.png"
  ];
  
  let scaleTemplate = ` <div class="scales">
            <span> {0} </span>
            <input type="radio" name="{1}" value="1" />
            <input type="radio" name="{1}" value="2" />
            <input type="radio" name="{1}" value="3" />
            <input type="radio" name="{1}" value="4" />
            <input type="radio" name="{1}" value="5" />
            <span> {2} </span>
          </div>`;
  
  let scaleMetadata = [
    {
      yin: "Masculine",
      yang: "Feminine",
      variableName: "gender"
    },
    {
      yin: "Childlike",
      yang: "Mature",
      variableName: "MatureChildlike"
    },
    {
      yin: "Unfriendly",
      yang: "Friendly",
      variableName: "friendly"
    },
    {
      yin: "Unintelligent",
      yang: "Intelligent",
      variableName: "Intelligent"
    }
  ];
  
  
  /*This function iterates over robotInfo and creates a button in
  the nav div corresponding to each robot*/
  
  function createOnlineQuestionnaire() {
    //let robotNames = Object.keys(imgInfo);
    let i = Math.floor(Math.random() * imgInfo.length);
    changeRobot(imgInfo[i]);
    changeScalesOrder();
  }
  
  /*The createRobotButtons should be called when the page has loaded*/
  
  // var body = document.getElementsByTagName("body")[0];
  // body.addEventListener("load", createRobotButtons(), false);
  
  /*This function will be called when a button is clicked and it will
  update the information displayed in the 'robot' div based on the 
  data stored within robotInfo*/
  
  function changeRobot(robotFaceSrc) {
    let robotDiv = document.getElementById("robotFace");
    robotDiv.src = robotFaceSrc;  
  }
  

  
  function changeScalesOrder() {
    
    let parentContainer = document.getElementsByClassName('section2')[0];  
    
    let shuffledMetadata = shuffle(scaleMetadata);
    
    for (let i = 0 ; i < shuffledMetadata.length; i ++) {
      let metadata = shuffledMetadata[i];
      let color = '';
      if (i % 2 == 0) {
        color = 'red';
      } else {
        color = 'blue';
      }
      let html = scaleTemplate.replace('{0}', metadata.yin)
                 .replace('{2}', `<span style="color: ${color}">${metadata.yang}</span>`)
                 .replace(/\{1\}/g, metadata.variableName);
      parentContainer.innerHTML += html;
    }  
  }
