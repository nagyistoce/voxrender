/// <reference path="~/Scripts/Support/_references.js" />

// :TODO: Document header here

// ----------------------------------------------------------------------------
//  Container managing a render. Performs communication with the server for
//  loading a scene and controlling image streaming and control feedback.
// ----------------------------------------------------------------------------
function VoxScene(id, name, file) {
    /// <summary>Initializes a new scene object</summary>

    this.id      = id;
    this._offset = { x: 0, y: 0 };
    this._name   = name;

    // Configure the render frame cache
    // :TODO: Initialize the image to a loading icon
    this.baseImage = new Image();
    this.baseImage.onload = $.proxy(function () {
        $(this).trigger("onBaseLoad", { img: this.baseImage });
    }, this);

    // Upload the file to the server if provided
    if (file) {
        var reader = new FileReader();
        reader.onloadend = $.proxy(function () {

            // Upload the scene to the render server
            $.post('api/scene/post',
                { id: file.name, data: reader.result },
                $.proxy(function (data, textStatus, jqXHR) {
                    // :TODO: Get from server -- prevent duplicate names;
                }, this), "json")
                .fail(function (jqXHR, textStatus, err) {
                    $(this).trigger("onUploadError", err);
                });

        }, this);
        reader.readAsDataURL(file);
    }
}

VoxScene.prototype =
{
    file: function () {
        /// <summary>Returns the name of the scene file on the server</summary>

        return this._name;
    },

    update: function (newImageData) {
        /// <summary>Processes the most recent frame from the server</summary>

        if (this.baseImage) this.baseImage.src = newImageData;
    },

    setPosition: function (x, y) {
        /// <summary>Changes the position of the image</summary>
        this._offset.x = x;
        this._offset.y = y;
        $(this).trigger('positionChanged', { image: this });
    },

    move: function (xDist, yDist) {
        /// <summary>Changes the position of the image</summary>
        this._offset.x += xDist;
        this._offset.y += yDist;
        $(this).trigger('positionChanged', { image: this });
    },

    zoom: function (factor) {
        /// <summary>Changes the zoomlevel of the image</summary>
        var scale = this._zoomLevel * factor;
        scale = Math.min(100, Math.max(0.1, scale));
        this.setZoom(scale);
    },

    setZoom: function (scale) {
        /// <summary>Changes the zoomlevel of the image</summary>
        this._zoomLevel = scale;
        $(this).trigger('zoomChanged', { image: this });
    },

    updateThumbnail: function () {
        /// <summary>Updates the thumbnail image in the sidebar</summary>

        var uiElem = $("#scene-" + this.id);
        var thumb = uiElem.find("img");
        thumb.attr('src', this.baseImage.src);
    },

    createUiElement: function () {
        /// <summary>Creates an HTML list element for the image bar</summary>

        var image = this;
        var id    = "fundusRadio" + this.id;

        // Create the HTML element for the image bar
        var elem = $("<div id='scene-" + image.id + "' style='width:100%; margin-bottom:5px;'>" +
                        "<img draggable='false' class='image-thumb'></img>" +
                        "<div style='height:32px; width:80%; left:10%; position:relative;'>" +
                           "<div style='float:left'>" + this.file() + "</div>" +
                           "<div style='float:right'; class='image-rem'>X</div>" +
                        "</div>" +
                     "</div>"
                     );

        // Setup the image to trigger display in the canvas
        var img = elem.find('.image-thumb');

        // Switches the display over the this scene
        img.click($.proxy(function () {
            // Check if this is already the current image
            if (WebPage.canvas.getScene() != this) {
                WebPage.canvas.setScene(this);
            }
        }, this));

        // Removes this scene from the side panel
        elem.find('.image-rem').click(function () {
            WebPage.imageBar.remove(image.id);
        });

        // Set the image for the thumbnail
        if (this.baseImage == null) {
            $(this).bind("onBaseLoad.ImageBar", function (event, data) {
                img.attr('src', this.baseImage.src);
                $(this).unbind("onBaseLoad.ImageBar");
            });
        }
        else img.attr('src', this.baseImage.src);

        return elem;
    },

    id:        0,    /// <field name='id'        type='Number'>Unique identifier</field>
    baseImage: null, /// <field name='baseImage' type='URI'>The original image dataURI, or null if unloaded</field>

    // *** Display Parameters ***
    _offset: { x: 0, y: 0 },
    _zoomLevel: 1.0,            /// <field name='_zoomLevel' type='Number'>Zoom scale for the image</field>
    _name: null,                /// <field name='_name'      type='String'>The scene file name on the server</field>
}