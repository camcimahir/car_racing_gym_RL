(function () {
  "use strict";

  // ── PDF Modal ──
  const modal      = document.getElementById("pdf-modal");
  const modalTitle = document.getElementById("modal-title");
  const modalFrame = document.getElementById("modal-iframe");
  const modalClose = document.getElementById("modal-close");

  function openPdf(pdfPath, title) {
    if (!modal || !modalFrame) return;
    modalTitle.textContent = title || "Report";
    modalFrame.src = encodeURI(pdfPath);
    modal.classList.add("active");
    document.body.style.overflow = "hidden";
  }

  function closePdf() {
    if (!modal) return;
    modal.classList.remove("active");
    document.body.style.overflow = "";
    setTimeout(function () { modalFrame.src = ""; }, 250);
  }

  document.querySelectorAll("[data-pdf]").forEach(function (btn) {
    btn.addEventListener("click", function (e) {
      e.preventDefault();
      openPdf(btn.getAttribute("data-pdf"), btn.getAttribute("data-title"));
    });
  });

  if (modalClose) modalClose.addEventListener("click", closePdf);
  if (modal) modal.addEventListener("click", function (e) {
    if (e.target === modal) closePdf();
  });
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") closePdf();
  });
})();
