/// <reference path="~/Scripts/_references.js" />
/// <reference path="~/Scripts/Page.js" />

// :TODO: Document header here

MessageType = {
    Warning: 0,
    Error: 1,
    Info: 2,
};

function Message(text, type, time) {
    /// <summary>Issues a new message to the user and appends it to the log</summary>
    /// <param name="text" type="String">String or element to use for the message content</param>
    /// <param name="type" type="MessageType">The type of the message</param>
    /// <param name="time" type="Number">The display time (in milliseconds) for the message</param>

    var messageBox = $("#messageBox");

    // Change the styling based on the message type
    switch (type) {
        // :TODO:
    };

    messageBox.empty().append(text);

    messageBox.fadeIn("fast", "swing");

    setTimeout(function () {
        messageBox.fadeOut("fast", "swing");
    }, (time == undefined) ? 500 : time);
};
