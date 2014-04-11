/// <reference path="~/Scripts/Support/_references.js" />

var VoxRender = VoxRender || {};

VoxRender.Server = {

    msgBegStream: function (sceneId) {
        WebPage.socket.send("\x01"+sceneId);
    },
    
    msgEndStream: function () {
        WebPage.socket.send("\x02");
    }
};
