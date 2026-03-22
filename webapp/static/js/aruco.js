document.addEventListener("DOMContentLoaded", () => {
    const btnGenerate = document.getElementById("btnGenerate");
    const btnDownload = document.getElementById("btnDownload");
    const btnPrint = document.getElementById("btnPrint");
    const markerIdInput = document.getElementById("markerId");
    const markerSizeSelect = document.getElementById("markerSize");
    const previewCard = document.getElementById("previewCard");
    const arucoImage = document.getElementById("arucoImage");
    const previewId = document.getElementById("previewId");

    let currentImageData = null;
    let currentId = 0;
    let currentSizeCm = 6;

    btnGenerate.addEventListener("click", async () => {
        const id = parseInt(markerIdInput.value, 10);
        currentSizeCm = parseInt(markerSizeSelect.value, 10);

        if (isNaN(id) || id < 0 || id > 49) {
            alert("Please enter a marker ID between 0 and 49.");
            return;
        }

        btnGenerate.disabled = true;
        btnGenerate.textContent = "Generating...";

        try {
            const resp = await fetch(`/api/generate-aruco?id=${id}&size=400`);
            const data = await resp.json();

            currentImageData = data.image;
            currentId = data.id;

            arucoImage.src = "data:image/png;base64," + data.image;
            previewId.textContent = data.id;
            previewCard.style.display = "block";
        } catch (err) {
            alert("Failed to generate marker: " + err.message);
        } finally {
            btnGenerate.disabled = false;
            btnGenerate.innerHTML = `
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="12" y1="8" x2="12" y2="16"/><line x1="8" y1="12" x2="16" y2="12"/></svg>
                Generate Marker`;
        }
    });

    btnDownload.addEventListener("click", () => {
        if (!currentImageData) return;

        const link = document.createElement("a");
        link.href = "data:image/png;base64," + currentImageData;
        link.download = `aruco_6x6_50_id${currentId}_${currentSizeCm}cm.png`;
        link.click();
    });

    btnPrint.addEventListener("click", () => {
        if (!currentImageData) return;

        // Create a hidden print area
        let printArea = document.querySelector(".print-area");
        if (printArea) printArea.remove();

        printArea = document.createElement("div");
        printArea.className = "print-area";
        printArea.style.display = "none";

        const img = document.createElement("img");
        img.src = "data:image/png;base64," + currentImageData;
        img.style.width = currentSizeCm + "cm";
        img.style.height = currentSizeCm + "cm";

        const label = document.createElement("div");
        label.className = "print-label";
        label.textContent = `ArUco 6x6_50 | ID: ${currentId} | ${currentSizeCm} cm`;

        printArea.appendChild(img);
        printArea.appendChild(label);
        document.querySelector(".app-container").appendChild(printArea);

        window.print();
    });

    // Generate on Enter key
    markerIdInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") btnGenerate.click();
    });
});
