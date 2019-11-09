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
            {2}
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

    // console.log(robotDiv);

    // console.log($('#robotFace'));

    // console.log(_(1234));
    robotDiv.src = robotFaceSrc;  
  }

  function test() {
    const $jqResult = $('.scales span:first-child');

    // $jqResult
    // .html((index, oldHtml) => {
    //   return `Baobao ${oldHtml} Lulu ${index}`
    // })
    // .css({
    //   'color': 'gold',
    //   // 'font-size': '20px', 
    //   'padding-left': '15px'
    // });


    // $('span').remove();

    const $span = $('<span>123</span>');
    
    $span.insertBefore($jqResult);
  }

  
  function changeScalesOrderBackup1() {
    
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

      const $templateElement = $(`
          <div class="scales">
            <span></span>
            <input type="radio" value="1" />
            <input type="radio" value="2" />
            <input type="radio" value="3" />
            <input type="radio" value="4" />
            <input type="radio" value="5" />
            <span></span>
          </div>
      `);

      const $spans = $templateElement.find('span');
      $templateElement.find('input').attr('name', metadata.variableName);

      $spans.eq(0).html(metadata.yin);
      $spans.eq(-1).html(metadata.yang).css('color', color);

      const $container = $(parentContainer);

      $container.append($templateElement);
    }  
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

      const $input = $('<input/>').attr({
        'type': 'radio',
        'name': metadata.variableName,
        'value': '1'
      });

      const $templateElement = $('<div/>').addClass('scales')
      .append($('<span/>').html(metadata.yin))
      .append($input)
      .append($input.clone().attr('value', '2'))
      .append($input.clone().attr('value', '3'))
      .append($input.clone().attr('value', '4'))
      .append($input.clone().attr('value', '5'))
      .append($('<span/>').html(metadata.yang).css('color', color));

      const $container = $(parentContainer);

      $container.append($templateElement);
    }  
  }



 