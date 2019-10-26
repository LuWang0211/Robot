function shuffle(array) {
    let shuffled = [];
    while (array.length > 0) {
      let randomNumber = Math.floor(Math.random() * array.length);
  
      let individual = array[randomNumber];
      array.splice(randomNumber, 1);    
      shuffled.push(individual);    
    }
    
    return shuffled;
  }