document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("sentimentForm");
    const modal = document.getElementById("resultModal");
    const closeModal = document.getElementById("closeModal");
    const resultText = document.getElementById("resultText");

    // Handle form submission
    form.addEventListener("submit", async(event) => {
        event.preventDefault();

        const comment = document.getElementById("comment").value;

        // Make API request
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ comment }),
        });

        const data = await response.json();

        // Update modal with result
        resultText.textContent = `Predicted Sentiment: ${data.predicted_sentiment}`;
        modal.style.display = "block";
    });

    // Close modal
    closeModal.addEventListener("click", () => {
        modal.style.display = "none";
    });

    // Close modal if clicking outside of it
    window.addEventListener("click", (event) => {
        if (event.target === modal) {
            modal.style.display = "none";
        }
    });
});