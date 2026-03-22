document.addEventListener("DOMContentLoaded", () => {
    const socket = io();

    // DOM elements
    const videoFeed = document.getElementById("videoFeed");
    const videoPlaceholder = document.getElementById("videoPlaceholder");
    const countdownOverlay = document.getElementById("countdownOverlay");
    const countdownNumber = document.getElementById("countdownNumber");

    const btnStart = document.getElementById("btnStart");
    const btnStop = document.getElementById("btnStop");
    const btnCapture = document.getElementById("btnCapture");
    const btnDismiss = document.getElementById("btnDismiss");

    const statusCamera = document.getElementById("statusCamera");
    const statusPerson = document.getElementById("statusPerson");
    const statusAruco = document.getElementById("statusAruco");
    const statusDistance = document.getElementById("statusDistance");
    const warningBar = document.getElementById("warningBar");
    const warningText = document.getElementById("warningText");

    const resultCard = document.getElementById("resultCard");
    const resultImage = document.getElementById("resultImage");
    const resultHeight = document.getElementById("resultHeight");
    const resultPixels = document.getElementById("resultPixels");
    const resultDistance = document.getElementById("resultDistance");

    let isStreaming = false;
    let countdownTimer = null;

    // --- Button Handlers ---
    btnStart.addEventListener("click", () => {
        socket.emit("start_stream");
        btnStart.disabled = true;
        statusCamera.innerHTML = '<span class="dot dot-yellow"></span> Connecting...';
        videoPlaceholder.querySelector("p").textContent = "Connecting to webcam...";
    });

    btnStop.addEventListener("click", () => {
        socket.emit("stop_stream");
        stopUI();
    });

    btnCapture.addEventListener("click", () => {
        startCountdown();
    });

    btnDismiss.addEventListener("click", () => {
        resultCard.style.display = "none";
    });

    // --- Countdown ---
    function startCountdown() {
        btnCapture.disabled = true;
        let count = 5;
        countdownNumber.textContent = count;
        countdownOverlay.style.display = "flex";

        countdownTimer = setInterval(() => {
            count--;
            if (count > 0) {
                countdownNumber.textContent = count;
            } else {
                clearInterval(countdownTimer);
                countdownOverlay.style.display = "none";
                socket.emit("capture");
            }
        }, 1000);
    }

    // --- Socket Events ---
    socket.on("stream_started", () => {
        isStreaming = true;
        videoFeed.style.display = "block";
        videoPlaceholder.style.display = "none";
        btnStart.disabled = true;
        btnStop.disabled = false;
        statusCamera.innerHTML = '<span class="dot dot-green"></span> Live';
    });

    socket.on("frame", (data) => {
        videoFeed.src = "data:image/jpeg;base64," + data.image;

        const status = data.status;

        // Person status
        if (status.person_detected) {
            if (status.is_cut_off) {
                statusPerson.innerHTML = '<span class="dot dot-yellow"></span> Partial';
            } else {
                statusPerson.innerHTML = '<span class="dot dot-green"></span> Detected';
            }
        } else {
            statusPerson.innerHTML = '<span class="dot dot-gray"></span> Not found';
        }

        // ArUco status
        if (status.aruco_detected) {
            statusAruco.innerHTML = '<span class="dot dot-green"></span> Detected';
            statusDistance.textContent = status.distance + " cm";
        } else {
            statusAruco.innerHTML = '<span class="dot dot-gray"></span> Not found';
            statusDistance.textContent = "--";
        }

        // Warning
        if (status.warning) {
            warningBar.style.display = "flex";
            warningText.textContent = status.warning;
        } else {
            warningBar.style.display = "none";
        }

        // Enable capture when person is fully visible
        const canCapture = status.person_detected && !status.is_cut_off;
        btnCapture.disabled = !canCapture;
    });

    socket.on("capture_result", (data) => {
        btnCapture.disabled = false;

        if (data.success) {
            resultImage.src = "data:image/jpeg;base64," + data.image;
            resultHeight.textContent = data.height_cm + " cm";
            resultPixels.textContent = data.pixel_height + " px";
            resultDistance.textContent = data.distance_cm + " cm";
            resultCard.style.display = "block";
        } else {
            alert("Capture failed: " + data.error);
        }
    });

    socket.on("stream_stopped", () => {
        stopUI();
    });

    socket.on("error", (data) => {
        videoPlaceholder.querySelector("p").textContent = data.message;
        statusCamera.innerHTML = '<span class="dot dot-red"></span> Error';
        btnStart.disabled = false;
        btnStop.disabled = true;
        btnCapture.disabled = true;
    });

    socket.on("disconnect", () => {
        stopUI();
    });

    function stopUI() {
        isStreaming = false;
        videoFeed.style.display = "none";
        videoPlaceholder.style.display = "flex";
        videoPlaceholder.querySelector("p").textContent = 'Click "Start Camera" to begin';
        countdownOverlay.style.display = "none";
        if (countdownTimer) clearInterval(countdownTimer);

        btnStart.disabled = false;
        btnStop.disabled = true;
        btnCapture.disabled = true;

        statusCamera.innerHTML = '<span class="dot dot-red"></span> Offline';
        statusPerson.innerHTML = '<span class="dot dot-gray"></span> --';
        statusAruco.innerHTML = '<span class="dot dot-gray"></span> --';
        statusDistance.textContent = "--";
        warningBar.style.display = "none";
    }
});
