/// <reference path="~/Scripts/Support/_references.js" />
/// <reference path="~/Scripts/VoxScene.js" />
/// <reference path="~/Scripts/Server.js" />

// ----------------------------------------------------------------------------
//  Canvas tool type enumeration
// ----------------------------------------------------------------------------
CanvasTool = {
    cursor: 0,  ///< Image drag tool
    brush:  1,  ///< Paint brush tool
    zoom:   2,  ///< Zoom in/out with drag
    text:   3,  ///< Insert text label
    range:  4
};

// ----------------------------------------------------------------------------
//  Creates a canvas element for displaying and annotating an image
// ----------------------------------------------------------------------------
function Canvas(canvasElem) {
    this._canvasElem = canvasElem;

    // Handle mouse events on the canvas
    var canvas = $(canvasElem);
    canvas.on('mousemove', $.proxy(this._onMouseMove, this));
    canvas.on('mousedown', $.proxy(function (e) {
        this._mousePos.x = e.screenX;
        this._mousePos.y = e.screenY;
        this._mousebutton[e.which-1] = true; }, this));
    canvas.on('mouseup', $.proxy(function (e) {
        this._mousebutton[e.which-1] = false;
    }, this));
    
    // Get the canvas offset for event position detection
    this._offset = canvas.offset();

    // Initialize the Hammer gesture detection library
    if (Modernizr.touch) {
        var hammertime = Hammer(canvasElem, {
            transform_always_block: true,
            transform_min_scale: 1,
            drag_block_horizontal: true,
            drag_block_vertical: true,
            drag_min_distance: 0
        });

        hammertime.on('touch drag transform hold', function (ev) {
            if (!WebPage.canvas._scene) return;
            switch (ev.type) {
                case 'touch':
                    WebPage.canvas.last_scale = WebPage.canvas._scene._zoomLevel;
                    WebPage.canvas.last_mx = 0;
                    WebPage.canvas.last_my = 0;
                    break;

                case 'drag':
                    var mx = WebPage.canvas.last_mx - ev.gesture.deltaX;
                    var my = WebPage.canvas.last_my - ev.gesture.deltaY;
                    WebPage.canvas._onMove(ev.gesture.srcEvent, mx, my);
                    WebPage.canvas.last_mx = ev.gesture.deltaX;
                    WebPage.canvas.last_my = ev.gesture.deltaY;
                    break;

                case 'transform':
                    var scale = WebPage.canvas.last_scale * ev.gesture.scale;
                    WebPage.canvas._scene.setZoom(scale);
                    break;

                case 'hold':
                    WebPage.contextMenu.open(ev.gesture.srcEvent);
                    break;
            }
        });
    }

    // Setup the canvas quick-tool buttons
    var quickTools = $("#quickTools");
    quickTools.buttonset();
    quickTools.find(".quickPageFit")
        .button({ icons: { primary: "icon-quick-fit" }, text: false })
        .click($.proxy(this.fitToPage, this));
    quickTools.find(".quickSave")
        .button({ icons: { primary: "icon-quick-save" }, text: false })
        .click($.proxy(this.download, this));
    quickTools.find(".quickPrint")
        .button({ icons: { primary: "icon-quick-print" }, text: false })
        .click($.proxy(this.print, this));
    quickTools.find(".quickZoomIn")
        .button({ icons: { primary: "icon-quick-zoom-in" }, text: false })
        .click($.proxy(function () { this.zoom(1.1); }, this));
    quickTools.find(".quickZoomOut")
        .button({ icons: { primary: "icon-quick-zoom-out" }, text: false })
        .click($.proxy(function () { this.zoom(1/1.1); }, this));

    // Hotkeys for image annotation and tools
    $(document).keypress(this, this._keyEventHandler);

    this._mousebutton = [false, false, false];
    this._mousePos    = { x: 0, y: 0 };
}

Canvas.prototype =
{
    setTool: function (tool) {
        this._tool = tool;
    },

    zoom: function (steps) {
        /// <summary>Changes the zoom level of the fundus image</summary>
        if (this._scene == null) return;
        var factor = Math.abs(steps);
        var zoom   = (steps > 0) ? factor : 1 / factor;
        this._scene.zoom(zoom);
    },

    fitToPage: function () {
        /// <summary>Fits the current fundus image, if present, to the page</summary>

        if (!this._scene || !this._scene.baseImage) return;

        var iw = this._scene.baseImage.width;
        var ih = this._scene.baseImage.height;
        var cw = this._canvasElem.width;
        var ch = this._canvasElem.height;

        var s = Math.min(cw/iw,ch/ih);
        
        this._blockRedraws = true;
        this._scene.setPosition(0, 0);
        this._scene.setZoom(s);
        this._blockRedraws = false;

        this.draw();
    },

    draw: function () {
        /// <summary>Redraws the image on the visible canvas element</summary>

        if (this._blockRedraws) return;

        // If there is no image set, clear the drawing canvas
        if (!this._scene) {
            var ctx = this._canvasElem.getContext("2d");
            ctx.save();
            ctx.fillStyle = "#7F7F7F";
            ctx.rect(0, 0, this._canvasElem.width, this._canvasElem.height);
            ctx.fill();
            var bgimg = document.getElementById("bgimg");
            ctx.drawImage(bgimg,
                (this._canvasElem.width - bgimg.width)   / 2,
                (this._canvasElem.height - bgimg.height) / 2);
            ctx.restore();
            return;
        }

        // Acquire a handle to the canvas context
        var ctx = this._canvasElem.getContext("2d");

        // Compute the image position offset
        var s = this._scene._zoomLevel;
        var w = this._canvasElem.width  / s / 2;
        var h = this._canvasElem.height / s / 2;
        var x = w + this._scene._offset.x;
        var y = h + this._scene._offset.y;
        x -= this._scene.baseImage.width / 2;
        y -= this._scene.baseImage.height / 2;

        // Draw the scene image
        ctx.save();
        ctx.imageSmoothingEnabled = false;
        ctx.fillStyle = "#7F7F7F";
        ctx.rect(0, 0, this._canvasElem.width, this._canvasElem.height);
        ctx.fill();
        ctx.scale(s, s);
        ctx.drawImage(this._scene.baseImage, x, y);
        ctx.restore();
    },

    setScene: function (scene, blockHistory) {
        /// <summary>Sets the scene to be displayed in the canvas</summary>
        /// <param name="image" type="VoxScene"></param>

        WebPage.toolBar.populate(scene);

        $(this._scene).unbind('.Canvas');

        if (this._scene) this._scene.updateThumbnail();

        this._scene = scene;

        // Detect a null image set so we can clear the view
        if (this._scene == null) {
            this.draw();
            VoxRender.Server.msgEndStream();
            return;
        }

        // Send the render start message to the server
        VoxRender.Server.msgBegStream(this._scene.file(), this._scene.id);

        var canvas = this;

        // Canvas interaction which does not modify the base image data (viewing)
        var onChange = $.proxy(this.draw, this);
        $(this._scene).bind('positionChanged.Canvas', onChange);
        $(this._scene).bind('zoomChanged.Canvas', onChange);
        $(this._scene).bind('displayChanged.Canvas', onChange);
        $(this._scene).bind('dataChanged.Canvas', onChange);
        $(this._scene).bind('onBaseLoad.Canvas', onChange);

        canvas.draw();
    },

    download: function () {
        /// <summary>Downloads the annotated image to the client</summary>

        // Setup the anchor element for image download
        var canvas = this._canvasElem;
        var a = $("<a id='link' href='#'>Download</a>");
        a.on("click", $.proxy(function () {
            a.attr("href", this._scene.baseImage.src)
             .attr("download", "Render.png");
            this.draw();
        }, this));
            
        // DOM 2 Events for initiating the anchor link
        var dispatchMouseEvent = function (target, var_args) {
            var e = document.createEvent("MouseEvents");
            e.initEvent.apply(e, Array.prototype.slice.call(arguments, 1));
            target.dispatchEvent(e);
        };
        dispatchMouseEvent(a[0], 'mouseover', true, true);
        dispatchMouseEvent(a[0], 'mousedown', true, true);
        dispatchMouseEvent(a[0], 'click',     true, true);
        dispatchMouseEvent(a[0], 'mouseup',   true, true);
    },

    print: function () {
        /// <summary>Prints the canvas image</summary>

        popup = window.open();
        popup.document.write('<img src="' + this._scene.baseImage.src + '";></img>');
        popup.document.close();
        popup.print();
        popup.close();

        this.draw();
    },

    getScene: function () {
        return this._scene;
    },

// Private:

    _onMouseMove: function (e, ui) {
        /// <param name="e" type="JQuery.Event">event</param>

        e.preventDefault()

        if (!this._scene) return;

        var moveX = this._mousePos.x - e.screenX;
        var moveY = this._mousePos.y - e.screenY;
        
        if (this._mousebutton[0]) {
            this._onMove(e, moveX, moveY);
        }
        else if (this._mousebutton[1]) {
            this._onMove(e, moveX, moveY, CanvasTool.cursor);
        }

        this._mousePos.x = e.screenX;
        this._mousePos.y = e.screenY;
    },

    _onMove: function (e, mx, my, tool) {
        var action = tool ? tool : this._tool;
        switch (action) {
            case CanvasTool.cursor:
                this._scene.revolveCamera(mx, my);
                break;
            // :TODO:
                var scale = this._scene._zoomLevel;
                var moveX = mx/scale;
                var moveY = my/scale;
                this._scene.move(-moveX, -moveY);
                break;
            case CanvasTool.brush:
                break;
            case CanvasTool.zoom:
                var factor = Math.pow(2, my / 300);
                this._scene.zoom(factor);
                break;
            case CanvasTool.range:
                var window = mx / 500.0 + this._scene._window;
                var level  = my / 500.0 + this._scene._level;
                this._scene.setWindowLevel(window, level);
                break;
        }
    },

    _keyEventHandler: function (e) {
        /// <param name="e" type="JQuery.Event">event</param>
        var canvas = e.data;
        var fundus = canvas._scene;
    },

    // Private:
    _mousebutton: [],           /// <field name='_mousebutton' type='Array'>Tracking array for mouse button status</field>
    _mousePos: { x: 0, y: 0 },  /// <field name='_mousePos'>The mouse position for the most recent event</field>
    _touchPos: { x: 0, y: 0 },  /// <field name='_mousePos'>The touch position for the most recent event</field>
    _scene: null,               /// <field name='_scene' type='VoxScene'>The image currently in this canvas</field>
    _canvasElem: null,          /// <field name='_canvasElem' type='Canvas'>The HTML canvas elemented associated with this object</field>
    _blockRedraws: false,       /// <field name='_blockRedraws' type='Boolean'>Blocks the draw function from being executed</field>
    _tool: CanvasTool.cursor,   /// <field name='_tool'></field>
}

// ----------------------------------------------------------------------------
//  Action for switching the currently displayed image
// ----------------------------------------------------------------------------
function SetSceneAction(oldImage, newImage) {
    this._old = oldImage;
    this._new = newImage;
}

SetSceneAction.prototype =
{
    text: "changing the active scene",

    // Private:

    undo: function () {
        WebPage.canvas.setScene(this._old, true);
    },

    redo: function () {
        WebPage.canvas.setScene(this._new, true);
    },

    _old: null, /// <field name='_index' type='VoxScene'>The previous scene</field>
    _new: null, /// <field name='_index' type='VoxScene'>The newer scene</field>
}