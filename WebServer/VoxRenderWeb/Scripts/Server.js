/// <reference path="~/Scripts/Support/_references.js" />

var VoxRender = VoxRender || {};

VoxRender.Server = {

    msgUpdateScene: function (scene) {

    },

    msgBegStream: function (file, id) {
        WebPage.socket.send("\x01" + file + "\x00" + id);
    },
    
    msgEndStream: function () {
        WebPage.socket.send("\x02");
    }
};
