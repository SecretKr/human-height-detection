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
    const btnModeAruco = document.getElementById("btnModeAruco");
    const btnModeCard = document.getElementById("btnModeCard");

    const statusCamera = document.getElementById("statusCamera");
    const statusPerson = document.getElementById("statusPerson");
    const statusAruco = document.getElementById("statusAruco");
    const statusDistance = document.getElementById("statusDistance");
    const statusCard = document.getElementById("statusCard");
    const statusArucoRow = document.getElementById("statusArucoRow");
    const statusDistanceRow = document.getElementById("statusDistanceRow");
    const statusCardRow = document.getElementById("statusCardRow");
    const warningBar = document.getElementById("warningBar");
    const warningText = document.getElementById("warningText");

    const resultCard = document.getElementById("resultCard");
    const resultImage = document.getElementById("resultImage");
    const resultHeight = document.getElementById("resultHeight");
    const resultPixels = document.getElementById("resultPixels");
    const resultDistance = document.getElementById("resultDistance");
    const resultSingle = document.getElementById("resultSingle");
    const resultMulti = document.getElementById("resultMulti");
    const resultPersonsList = document.getElementById("resultPersonsList");

    const instructionsAruco = document.getElementById("instructionsAruco");
    const instructionsCardMode = document.getElementById("instructionsCardMode");

    let isStreaming = false;
    let countdownTimer = null;
    let currentMode = "aruco";

    // --- Mode Switching ---
    function switchMode(mode) {
        currentMode = mode;

        btnModeAruco.classList.toggle("active", mode === "aruco");
        btnModeCard.classList.toggle("active", mode === "card");

        statusArucoRow.style.display = mode === "aruco" ? "" : "none";
        statusDistanceRow.style.display = mode === "aruco" ? "" : "none";
        statusCardRow.style.display = mode === "card" ? "" : "none";

        instructionsAruco.style.display = mode === "aruco" ? "" : "none";
        instructionsCardMode.style.display = mode === "card" ? "" : "none";

        if (mode === "aruco") {
            statusAruco.innerHTML = '<span class="dot dot-gray"></span> --';
            statusDistance.textContent = "--";
        } else {
            statusCard.innerHTML = '<span class="dot dot-gray"></span> --';
        }

        btnCapture.disabled = true;

        fetch("/api/set-mode", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ mode }),
        }).catch(console.error);
    }

    btnModeAruco.addEventListener("click", () => switchMode("aruco"));
    btnModeCard.addEventListener("click", () => switchMode("card"));

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
            const count = status.person_count || 1;
            if (status.is_cut_off) {
                statusPerson.innerHTML = '<span class="dot dot-yellow"></span> Partial';
            } else {
                const label = count > 1 ? `${count} detected` : "Detected";
                statusPerson.innerHTML = `<span class="dot dot-green"></span> ${label}`;
            }
        } else {
            statusPerson.innerHTML = '<span class="dot dot-gray"></span> Not found';
        }

        // Reference marker status
        if (status.mode === "aruco") {
            if (status.aruco_detected) {
                statusAruco.innerHTML = '<span class="dot dot-green"></span> Detected';
                statusDistance.textContent = status.distance + " cm";
            } else {
                statusAruco.innerHTML = '<span class="dot dot-gray"></span> Not found';
                statusDistance.textContent = "--";
            }
        } else if (status.mode === "card") {
            if (status.card_detected) {
                statusCard.innerHTML = '<span class="dot dot-green"></span> Detected';
            } else {
                statusCard.innerHTML = '<span class="dot dot-gray"></span> Not found';
            }
        }

        // Warning
        if (status.warning) {
            warningBar.style.display = "flex";
            warningText.textContent = status.warning;
        } else {
            warningBar.style.display = "none";
        }

        // Enable capture button based on mode
        let canCapture = status.person_detected && !status.is_cut_off;
        if (status.mode === "aruco") canCapture = canCapture && status.aruco_detected;
        if (status.mode === "card") canCapture = canCapture && status.card_detected;
        btnCapture.disabled = !canCapture;
    });

    socket.on("capture_result", (data) => {
        btnCapture.disabled = false;

        if (!data.success) {
            alert("Capture failed: " + data.error);
            return;
        }

        resultImage.src = "data:image/jpeg;base64," + data.image;

        if (data.mode === "card") {
            resultSingle.style.display = "none";
            resultMulti.style.display = "block";
            resultPersonsList.innerHTML = "";

            data.persons.forEach((p) => {
                const row = document.createElement("div");
                row.className = "person-result-row";
                row.innerHTML = `
                    <span class="person-label">Person ${p.person_idx}</span>
                    <span class="person-height">${p.height_cm} cm</span>
                    <span class="person-pixels">${p.pixel_height} px</span>
                `;
                resultPersonsList.appendChild(row);
            });
        } else {
            resultSingle.style.display = "block";
            resultMulti.style.display = "none";
            resultHeight.textContent = data.height_cm + " cm";
            resultPixels.textContent = data.pixel_height + " px";
            resultDistance.textContent = data.distance_cm + " cm";
        }

        resultCard.style.display = "block";
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
        statusCard.innerHTML = '<span class="dot dot-gray"></span> --';
        warningBar.style.display = "none";
    }
});
