document.addEventListener("DOMContentLoaded", () => {
    const socket = io();

    const btnStart = document.getElementById("btnDebugStart");
    const btnStop = document.getElementById("btnDebugStop");

    const debugImage = document.getElementById("debugImage");
    const debugBlur = document.getElementById("debugBlur");
    const debugHls = document.getElementById("debugHls");
    const debugEdges = document.getElementById("debugEdges");
    const debugClosed = document.getElementById("debugClosed");
    const debugOpened = document.getElementById("debugOpened");
    const debugContours = document.getElementById("debugContours");

    const statusDetected = document.getElementById("statusDetected");
    const statusRatio = document.getElementById("statusRatio");
    const statusArea = document.getElementById("statusArea");
    const statusScale = document.getElementById("statusScale");

    const controls = {
        blurKsize: document.getElementById("blurKsize"),
        hLow: document.getElementById("hLow"),
        hHigh: document.getElementById("hHigh"),
        lLow: document.getElementById("lLow"),
        lHigh: document.getElementById("lHigh"),
        sLow: document.getElementById("sLow"),
        sHigh: document.getElementById("sHigh"),
        cannyLow: document.getElementById("cannyLow"),
        cannyHigh: document.getElementById("cannyHigh"),
        morphKsize: document.getElementById("morphKsize"),
        closeIter: document.getElementById("closeIter"),
        openIter: document.getElementById("openIter"),
        minAreaRatio: document.getElementById("minAreaRatio"),
        aspectTolerance: document.getElementById("aspectTolerance"),
    };

    const outputs = {
        blurKsize: document.getElementById("blurKsizeValue"),
        hLow: document.getElementById("hLowValue"),
        hHigh: document.getElementById("hHighValue"),
        lLow: document.getElementById("lLowValue"),
        lHigh: document.getElementById("lHighValue"),
        sLow: document.getElementById("sLowValue"),
        sHigh: document.getElementById("sHighValue"),
        cannyLow: document.getElementById("cannyLowValue"),
        cannyHigh: document.getElementById("cannyHighValue"),
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
        outputs[id].textContent = value;
    }

    function getParams() {
        return {
            blur_ksize: parseInt(controls.blurKsize.value, 10),
            h_low: parseInt(controls.hLow.value, 10),
            h_high: parseInt(controls.hHigh.value, 10),
            l_low: parseInt(controls.lLow.value, 10),
            l_high: parseInt(controls.lHigh.value, 10),
            s_low: parseInt(controls.sLow.value, 10),
            s_high: parseInt(controls.sHigh.value, 10),
            canny_low: parseInt(controls.cannyLow.value, 10),
            canny_high: parseInt(controls.cannyHigh.value, 10),
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
        if (data.hls) debugHls.src = "data:image/jpeg;base64," + data.hls;
        if (data.edges) debugEdges.src = "data:image/jpeg;base64," + data.edges;
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
