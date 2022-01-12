var cities = [{
    city: "Kaohsiung City",
    name: "高雄市"
},
{
    city: "Pingtung",
    name: "屏東縣"
},
{
    city: "Tainan City",
    name: "臺南市"
},
{
    city: "Hsinchu City",
    name: "新竹市"
},
{
    city: "Hsinchu",
    name: "新竹縣"
},
{
    city: "Yilan",
    name: "宜蘭縣"
},
{
    city: "Keelung City",
    name: "基隆市"
},
{
    city: "Miaoli",
    name: "苗栗縣"
},
{
    city: "Taipei City",
    name: "臺北市"
},
{
    city: "New Taipei City",
    name: "新北市"
},
{
    city: "Taoyuan",
    name: "桃園市"
},
{
    city: "Changhua",
    name: "彰化縣"
},
{
    city: "Chiayi",
    name: "嘉義縣"
},
{
    city: "Chiayi City",
    name: "嘉義市"
},
{
    city: "Hualien",
    name: "花蓮縣"
},
{
    city: "Nantou",
    name: "南投縣"
},
{
    city: "Taichung City",
    name: "臺中市"
},
{
    city: "Yunlin",
    name: "雲林縣"
},
{
    city: "Taitung",
    name: "臺東縣"
},
{
    city: "Penghu",
    name: "澎湖縣"
},
{
    city: "Kinmen",
    name: "金門縣"
},
{
    city: "Lienchiang",
    name: "連江縣"
}
];

var vm = new Vue({
    el: "#app",
    data: {
        city: "tw",
        name: "臺灣"
    },
    computed: {
        set_city: function () {
            var vobj = this;
            var city = cities.find((element) => element.city == vobj.city);
            return city;
        }
    }
});

$("path").mouseenter(function () {
    var city = $(this).attr("name");
    vm.city = city;
    vm.name = vm.set_city.name;
});
$("path").click(function () {
    document.getElementById("iframe").src = "html/" + vm.name + ".html";
    window.scrollTo(window.scrollX, window.innerHeight);
})

function resize() {
    let twSvg = document.getElementById("tw_svg")
    twSvg.style.height = `${window.innerHeight - 157}px`
}

window.onload = () => {
    // console.log(`onresize: ${window.innerHeight}`);
    resize()
}

window.onresize = () => {
    // console.log(`onresize: ${window.innerHeight}`);
    resize()
}
