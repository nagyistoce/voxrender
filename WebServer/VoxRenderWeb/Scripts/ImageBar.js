/// <reference path="~/Scripts/Support/_references.js" />
/// <reference path="~/Scripts/Page.js" />
/// <reference path="~/Scripts/VoxScene.js" />

// :TODO: Document header here

// ----------------------------------------------------------------------------
//  Manages the sortable list of scenes in the application panel
// ----------------------------------------------------------------------------
function ImageBar() {
    this.init();
};

ImageBar.prototype =
{
    init: function () {
        /// <summary>Initializes the application's image bar</summary>

        this._$element = $("#imageBar");
        this._scenes   = [];
    },

    add: function (scene, index, active, suppressHistory) {
        /// <summary>Adds a new scene to the scene bar</summary>
        /// <param name="scene" type="VoxScene">The scene to be added</param>
        /// <param name="index" type="Number">The index of insertion</param>
        /// <param name="active" type="Boolean">If true, the image is displayed in the view</param>
        /// <param name="suppressHistory" type="Boolean">If true, action will not be added to the history</param>

        var newIndex;
        if (index == undefined) newIndex = this._scenes.length;
        else newIndex = Math.min(this._scenes.length, index);

        // Create the HTML element for the image bar
        var sceneElem = scene.createUiElement();
        if (newIndex >= this._scenes.length) $("#imagePane").append(sceneElem);
        else $("#imagePane").children(":eq(" + (index-1) + ")").after(sceneElem);

        this._scenes.splice(newIndex, 0, scene);

        // Present this image in the current view
        var isActive = (active || active == undefined);
        if (isActive) WebPage.canvas.setScene(scene);

        if (!suppressHistory) WebPage.history.push(new AddRemSceneAction(scene, newIndex, false, isActive));
    },

    remove: function (id) {
        /// <summary>Removes an image from the bar by its id</summary>
        /// <param name="image" type="String">The id of the image to remove</param>

        var index = null;
        for (var i = 0; i < this._scenes.length; i++) {
            if (this._scenes[i].id == id) { index = i; break; }
        }
        if (index == null) return;

        this.removeByIndex(index);
    },

    removeByIndex: function (index, suppressHistory) {
        /// <summary>Removes an image from the bar by its index</summary>
        /// <param name="index" type="Number">The index of insertion</param>
        /// <param name="suppressHistory" type="Boolean">If true, not action will be added to the history</param>

        if (index > this._scenes.length) return;

        var scene = this._scenes[index];

        var active = false;
        if (WebPage.canvas.getScene() == scene) {
            WebPage.canvas.setScene(null);
            active = true;
        }

        $(scene).unbind(".ImageBar");

        $("#scene-" + scene.id).detach();
        this._scenes.splice(index, 1);

        if (!suppressHistory) WebPage.history.push(new AddRemSceneAction(
            scene, index, true, active));
    },

    getState: function () {
        /// <summary>Returns the compressed state of the image bar</summary>

        return { images: this._scenes };
    },

    reset: function () {
        /// <summary>Resets the image bar to its default state</summary>

        $("#imagePane").empty();
        this._scenes = [];
    },

    restore: function (state) {
        /// <summary>Restores the image bar from a compressed state</summary>

        for (var i = 0; i < state.images.length; i++)
            this.add(state.images[i], i, true);
    },

    // Private:

    _$element: null, /// <field name='_$element' type=''>JQuery handle to the image bars DOM element</field>
    _scenes:   null, /// <field name='_scenes' type='Array'>Images loaded into the application</field>
};

// ----------------------------------------------------------------------------
//  Action for adding a scene to the application's scene bar
// ----------------------------------------------------------------------------
function AddRemSceneAction(scene, index, isRemove, active) {
    this._index  = index;
    this._scene  = scene;
    this._active = active;
    this.undo = isRemove ? this._add : this._remove;
    this.redo = isRemove ? this._remove : this._add;
}

AddRemSceneAction.prototype =
{
    undo: null,
    redo: null,
    text: "Loading a new scene",

    // Private:

    _remove: function () {
        WebPage.imageBar.removeByIndex(this._index, true);
    },

    _add: function () {
        WebPage.imageBar.add(this._scene, this._index, this._active, true);
    },

    _active: false, /// <field name='_index' type='Boolean'>True if the image is in the view</field>
    _index: 0,      /// <field name='_index' type='Number'>The index of insertion</field>
    _scene: null,   /// <field name='_scene' type='VoxScene'>The image</field>
}