document.addEventListener("DOMContentLoaded", function () {
    // Automatically focus and smoothly scroll to the results if they exist
    const resultSection = document.getElementById("result");
    if (resultSection) {
        resultSection.scrollIntoView({ behavior: "smooth", block: "center" });
    }
});