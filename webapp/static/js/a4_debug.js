document.addEventListener("DOMContentLoaded", () => {
    const socket = io();

    const btnStart = document.getElementById("btnDebugStart");
    const btnStop = document.getElementById("btnDebugStop");

    const debugImage = document.getElementById("debugImage");
    const debugBlur = document.getElementById("debugBlur");
    const debugL = document.getElementById("debugL");
    const debugClahe = document.getElementById("debugClahe");
    const debugThresh = document.getElementById("debugThresh");
    const debugClosed = document.getElementById("debugClosed");
    const debugOpened = document.getElementById("debugOpened");
    const debugContours = document.getElementById("debugContours");

    const statusDetected = document.getElementById("statusDetected");
    const statusRatio = document.getElementById("statusRatio");
    const statusArea = document.getElementById("statusArea");
    const statusScale = document.getElementById("statusScale");

    const controls = {
        blurKsize: document.getElementById("blurKsize"),
        claheClip: document.getElementById("claheClip"),
        claheGrid: document.getElementById("claheGrid"),
        adaptBlock: document.getElementById("adaptBlock"),
        adaptC: document.getElementById("adaptC"),
        morphKsize: document.getElementById("morphKsize"),
        closeIter: document.getElementById("closeIter"),
        openIter: document.getElementById("openIter"),
        minAreaRatio: document.getElementById("minAreaRatio"),
        aspectTolerance: document.getElementById("aspectTolerance"),
    };

    const outputs = {
        blurKsize: document.getElementById("blurKsizeValue"),
        claheClip: document.getElementById("claheClipValue"),
        claheGrid: document.getElementById("claheGridValue"),
        adaptBlock: document.getElementById("adaptBlockValue"),
        adaptC: document.getElementById("adaptCValue"),
        morphKsize: document.getElementById("morphKsizeValue"),
        closeIter: document.getElementById("closeIterValue"),
        openIter: document.getElementById("openIterValue"),
        minAreaRatio: document.getElementById("minAreaRatioValue"),
        aspectTolerance: document.getElementById("aspectToleranceValue"),
    };

    let updateTimer = null;
    let streaming = false;

    function formatFloat(value, digits) {
        return Number.parseFloat(value).toFixed(digits);
    }

    function updateOutput(id, value) {
        if (!outputs[id]) return;
        if (id === "minAreaRatio") {
            outputs[id].textContent = formatFloat(value, 3);
            return;
        }
        if (id === "aspectTolerance") {
            outputs[id].textContent = formatFloat(value, 2);
            return;
        }
        if (id === "claheClip") {
            outputs[id].textContent = formatFloat(value, 1);
            return;
        }
        outputs[id].textContent = value;
    }

    function getParams() {
        return {
            blur_ksize: parseInt(controls.blurKsize.value, 10),
            clahe_clip: parseFloat(controls.claheClip.value),
            clahe_grid: parseInt(controls.claheGrid.value, 10),
            adapt_block: parseInt(controls.adaptBlock.value, 10),
            adapt_c: parseInt(controls.adaptC.value, 10),
            morph_ksize: parseInt(controls.morphKsize.value, 10),
            close_iter: parseInt(controls.closeIter.value, 10),
            open_iter: parseInt(controls.openIter.value, 10),
            min_area_ratio: parseFloat(controls.minAreaRatio.value),
            aspect_tolerance: parseFloat(controls.aspectTolerance.value),
        };
    }

    function scheduleUpdate() {
        if (!streaming) return;
        if (updateTimer) clearTimeout(updateTimer);
        updateTimer = setTimeout(() => {
            socket.emit("update_a4_debug", getParams());
        }, 120);
    }

    Object.keys(controls).forEach((key) => {
        const input = controls[key];
        updateOutput(key, input.value);
        input.addEventListener("input", () => {
            updateOutput(key, input.value);
            scheduleUpdate();
        });
    });

    btnStart.addEventListener("click", () => {
        btnStart.disabled = true;
        socket.emit("start_a4_debug", getParams());
    });

    btnStop.addEventListener("click", () => {
        socket.emit("stop_a4_debug");
    });

    socket.on("a4_debug_started", () => {
        streaming = true;
        btnStart.disabled = true;
        btnStop.disabled = false;
    });

    socket.on("a4_debug_stopped", () => {
        streaming = false;
        btnStart.disabled = false;
        btnStop.disabled = true;
    });

    socket.on("a4_debug_frame", (data) => {
        if (data.overlay) debugImage.src = "data:image/jpeg;base64," + data.overlay;
        if (data.blur) debugBlur.src = "data:image/jpeg;base64," + data.blur;
        if (data.l_channel) debugL.src = "data:image/jpeg;base64," + data.l_channel;
        if (data.clahe) debugClahe.src = "data:image/jpeg;base64," + data.clahe;
        if (data.thresh) debugThresh.src = "data:image/jpeg;base64," + data.thresh;
        if (data.closed) debugClosed.src = "data:image/jpeg;base64," + data.closed;
        if (data.opened) debugOpened.src = "data:image/jpeg;base64," + data.opened;
        if (data.contours) debugContours.src = "data:image/jpeg;base64," + data.contours;

        if (data.status && data.status.detected) {
            statusDetected.textContent = "Detected";
            statusRatio.textContent = data.status.ratio ?? "--";
            statusArea.textContent = data.status.area_ratio ?? "--";
            statusScale.textContent = data.status.cm_per_px ? data.status.cm_per_px + " cm/px" : "--";
        } else {
            statusDetected.textContent = "Not detected";
            statusRatio.textContent = "--";
            statusArea.textContent = "--";
            statusScale.textContent = "--";
        }
    });

    socket.on("a4_debug_error", (data) => {
        streaming = false;
        btnStart.disabled = false;
        btnStop.disabled = true;
        alert(data.message || "A4 debug error");
    });

    socket.on("disconnect", () => {
        streaming = false;
        btnStart.disabled = false;
        btnStop.disabled = true;
    });
});
