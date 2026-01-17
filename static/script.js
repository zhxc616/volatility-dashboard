document.addEventListener('DOMContentLoaded', function() {
    console.log("JavaScript is loaded and running!");

    async function fetchVolatility() {
        // find the elements needed
        const boxElement = document.getElementById('volatility-box');
        const displayElement = document.getElementById('volatility-display');

        if (!boxElement) return;

        // get the ticker from the HTML data attribute
        const ticker = boxElement.getAttribute('data-ticker');
        console.log("Fetching volatility for:", ticker);

        try {
            // call api
            const response = await fetch(`/api/get-volatility?ticker=${ticker}`);
            const data = await response.json();

            // update page
            if (data.volatility) {
                displayElement.innerText = data.volatility + "%";

                // colour changes based on risk level
                if (data.volatility > 40) {
                    displayElement.style.color = "#FF3B30";
                } else {
                    displayElement.style.color = "#00B4F0";
                }
            } else {
                displayElement.innerText = "N/A";
            }

        } catch (error) {
            console.error("Error fetching volatility:", error);
            displayElement.innerText = "Error";
        }
    }

    fetchVolatility();
});