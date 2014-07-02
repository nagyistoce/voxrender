/// <reference path="~/Scripts/Support/_references.js" />

var VoxRender = VoxRender || {};

VoxRender.Server = {

    msgModify: function (transform, options) {
        /// <summary>Request to apply a scene modifier</summary>
        var data = transform;
        if (options) data += JSON.stringify(options);
        WebPage.socket.send("\x04" + data);
    },

    msgBegStream: function (file, id) {
        /// <summary>Request to begin rendering a scenefile</summary>
        WebPage.socket.send("\x01" + file + "\x00" + id);
    },

    msgUpdate: function (scene) {
        /// <summary>Updates the scene file being rendered</summary>
        WebPage.socket.send("\x06" + JSON.stringify(scene))
    },

    msgEndStream: function () {
        /// <summary>Request to discontinue rendering a scene</summary>
        WebPage.socket.send("\x02");
    }
};
