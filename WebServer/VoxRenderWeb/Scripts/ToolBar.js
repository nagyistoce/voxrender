/// <reference path="~/Scripts/Support/_references.js" />
/// <reference path="~/Scripts/Page.js" />
/// <reference path="~/Scripts/VoxScene.js" />

// ----------------------------------------------------------------------------
//  Manages the editing and annotation tools for the fundus images
// ----------------------------------------------------------------------------
function ToolBar() {
    this.init();
};

ToolBar.prototype =
{
    init: function () {
        /// <summary>Initializes the application's image bar</summary>

        $("#toolsPane").hide();
        
        this._$element = $("#toolBar");
        this._image = null;
    },

    populate: function (scene) {
        /// <summary>Populates the toolbar controls with the specified image's data</summary>

        this._scene = scene;

        if (this._scene == null) { return; }
    },

    disable: function () {
        /// <summary>Disables the controls on the toolbar</summary>

    },

    // Private:

    _$element: null, /// <field name='_$element' type=''>JQuery handle to the image bars DOM element</field>
    _scene: null,    /// <field name='_scene' type='VoxScene'>Scene currently associated with the tools</field>
};