//pobieranie elementów z html
//const navElement1 = "." + document.querySelector("#aboutUs").id;


const navigacja = document.querySelector(" nav .nav");
const elementyLi = Array.prototype.slice.apply(document.querySelectorAll('.nav li'));
for (let i = 0; i < elementyLi.length; i++) {
    elementyLi[i].addEventListener('click', function () {
        //this.console.log('działa');
        const idElementu = elementyLi[i].id;
        //console.log(idElementu);
        window.scroll(0, document.querySelector(`[data-Toscroll=${idElementu}]`).offsetTop - document.querySelector('nav').offsetHeight);

    })

};



//const navElements = $('nav a');
// elementy nawigacji
//let goToElement = '';
//
//navigacja.addEventListener("click", function () {
// console.log(navElement1);
//const goToElement = document.querySelector(navElement1).offsetTop;
//console.log(goToElement);
//window.scrollTo(0, goToElement);
// window.scrollTo(0, 1500);
//})
document.querySelector('.srollTop').addEventListener('click', function () {
    window.scrollTo(0, 0);
    console.log("działa")
});
document.querySelector('.aboutUs .button').addEventListener('click', function () {
    window.scrollTo(0, document.querySelector(".portfolio").offsetTop - document.querySelector('nav').offsetHeight);
});
document.querySelector('.ourSkill .button').addEventListener('click', function () {
    window.scrollTo(0, document.querySelector(".contact").offsetTop - document.querySelector('nav').offsetHeight);
});
// events

/*$('nav li').on('click', function () {
    const clickedElement = "." + $(this).attr('id');
    console.log(clickedElement);
    $('body', 'html').animate({
        scrollTop: $(clickedElement).offset().top
    }, 1000)
})*/
/*navElements.on('click', function () {
    goToElement = "." + $(this).attr('id');
    const activeSection = document.querySelector(goToElement).offsetTop;
    window.scrollTo(0, 3000);
    console.log(activeSection);

})
document.addEventListener('scroll', function () {

})*/

//functions