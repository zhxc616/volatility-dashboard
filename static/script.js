document.addEventListener('DOMContentLoaded', function() {
    console.log("JavaScript is loaded and running!");

    async function fetchVolatility() {
        // 1. Find the elements we need
        const boxElement = document.getElementById('volatility-box');
        const displayElement = document.getElementById('volatility-display');

        // Safety check: Does the box exist? (It might not if there is an error on page)
        if (!boxElement) return;

        // 2. Get the ticker from the HTML data attribute
        const ticker = boxElement.getAttribute('data-ticker');
        console.log("Fetching volatility for:", ticker);

        try {
            // 3. Call the Python API
            const response = await fetch(`/api/get-volatility?ticker=${ticker}`);
            const data = await response.json();

            // 4. Update the page
            if (data.volatility) {
                displayElement.innerText = data.volatility + "%";

                // Optional: Change colour based on risk level
                if (data.volatility > 40) {
                    displayElement.style.color = "#FF3B30"; // Red for high risk
                } else {
                    displayElement.style.color = "#00B4F0"; // Blue for normal
                }
            } else {
                displayElement.innerText = "N/A";
            }

        } catch (error) {
            console.error("Error fetching volatility:", error);
            displayElement.innerText = "Error";
        }
    }

    // Run the function
    fetchVolatility();
});