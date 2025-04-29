document.addEventListener("DOMContentLoaded", function () {
  // Theme switching functionality
  const themeToggle = document.querySelector(
    '.theme-switch input[type="checkbox"]'
  );
  const currentTheme = localStorage.getItem("theme");

  // Check for saved theme preference or respect OS preference
  if (currentTheme) {
    document.body.classList.add(currentTheme);
    if (currentTheme === "dark-theme") {
      themeToggle.checked = true;
    }
  } else {
    // Check if user prefers dark mode in their OS settings
    if (
      window.matchMedia &&
      window.matchMedia("(prefers-color-scheme: dark)").matches
    ) {
      document.body.classList.add("dark-theme");
      themeToggle.checked = true;
      localStorage.setItem("theme", "dark-theme");
    } else {
      document.body.classList.add("light-theme");
      localStorage.setItem("theme", "light-theme");
    }
  }

  // Theme switch event handler
  themeToggle.addEventListener("change", switchTheme, false);

  function switchTheme(e) {
    if (e.target.checked) {
      document.body.classList.replace("light-theme", "dark-theme");
      localStorage.setItem("theme", "dark-theme");
    } else {
      document.body.classList.replace("dark-theme", "light-theme");
      localStorage.setItem("theme", "light-theme");
    }
  }

  // Show results function for MCQs page
  const showResultsBtn = document.getElementById("show-results-btn");
  if (showResultsBtn) {
    showResultsBtn.addEventListener("click", function () {
      // Show correct answers with animation
      document.querySelectorAll(".correct-answer").forEach(function (answer) {
        answer.style.display = "block";
        answer.classList.add("fade-in");
      });

      // Show download button
      const downloadBtn = document.getElementById("download-pdf-btn");
      if (downloadBtn) {
        downloadBtn.style.display = "inline-block";
        downloadBtn.classList.add("fade-in");
      }
    });
  }

  // Form validation
  const form = document.querySelector("form");
  if (form) {
    form.addEventListener("submit", function (event) {
      const url = document.getElementById("url");
      const manualText = document.getElementById("manual_text");
      const files = document.getElementById("file");

      // Check if at least one input method is provided
      if (
        (!url || !url.value) &&
        (!manualText || !manualText.value) &&
        a(!files || files.files.length === 0)
      ) {
        event.preventDefault();

        // Create alert message if it doesn't exist
        if (!document.querySelector(".alert-warning")) {
          const alert = document.createElement("div");
          alert.className = "alert alert-warning mt-3";
          alert.textContent =
            "Please provide at least one input method: URL, text, or file upload.";
          form.prepend(alert);

          // Auto-dismiss alert after 5 seconds
          setTimeout(() => {
            alert.classList.add("fade");
            setTimeout(() => alert.remove(), 500);
          }, 5000);
        }
      }
    });
  }

  // Custom file input label update
  const fileInput = document.querySelector(".custom-file-input");
  if (fileInput) {
    fileInput.addEventListener("change", function (e) {
      let fileName = "";
      if (this.files && this.files.length > 1) {
        fileName = this.files.length + " files selected";
      } else {
        fileName = e.target.value.split("\\").pop();
      }

      const nextSibling = e.target.nextElementSibling;
      if (nextSibling && nextSibling.classList.contains("custom-file-label")) {
        nextSibling.textContent = fileName || "Choose file(s)";
      }
    });
  }

  // Add fade-in animation to page content
  document.querySelector(".main-container").classList.add("fade-in");
});
