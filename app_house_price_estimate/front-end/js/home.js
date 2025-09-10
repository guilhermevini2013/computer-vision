$(document).ready(function () {
    $.ajax({
        url: "http://localhost:5000/neighborhoods",
        method: "GET",
        success: function (data) {
            let select = $("#neighborhoods");

            data.all_neighborhoods.forEach(function(neighborhood) {
                select.append(new Option(neighborhood, neighborhood));
            });
        },
        error: function (xhr, status, error) {
            console.error("Error:", error);
        },
    });
});

$("#send").click(function() {
    let garage = $(".entrance input[placeholder='Number of parking spaces']").val();
    let area = $(".entrance input[placeholder='Area m²']").val();
    let bedrooms = $(".entrance2 input[placeholder='Bedrooms']").val();
    let bathrooms = $(".entrance2 input[placeholder='Bathrooms']").val();
    let neighborhood = $("#neighborhoods").val();

    let data = {
        area: area,
        bedroom: bedrooms,
        bathroom: bathrooms,
        garage: garage,
        neighborhood: neighborhood
    };

    $.ajax({
        url: "http://localhost:5000/preview-price",
        method: "POST",
        data: JSON.stringify(data),
        contentType: "application/json",
        success: function(response) {
            $("#result").text(`Estimated value: ${response.estimated_price}`);
        },
        error: function(xhr, status, error) {
            console.error("Error:", error);
            $("#result").text("Error calculating estimate. Please try again.");
        },
    });
});
