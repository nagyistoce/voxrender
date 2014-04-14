/// <reference path="~/Scripts/Support/_references.js" />

var VoxRender = VoxRender || {};

VoxRender.Server = {

    msgUpdateScene: function (scene) {

    },

    msgBegStream: function (sceneId) {
        WebPage.socket.send("\x01"+sceneId);
    },
    
    msgEndStream: function () {
        WebPage.socket.send("\x02");
    }
};
