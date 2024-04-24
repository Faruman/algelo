$( ".btn-delete" ).on( "click", function(e) {
    e.preventDefault();
    var file_metadata = $( this ).parents('.file-card').data("file");
    $.post( "/api/delete", {file_metadata: JSON.stringify(file_metadata), action: "delete"} )
    .done(function( data_id ) {
      $('div[data-id="'+data_id+'"]').parent().remove();
    })
    .fail(function() {
      alert( "Error: Unable to delete the selected file, please try again." );
    })
});

$(".dropdown-item").on( "click", function(e) {
    e.preventDefault();
    var file_metadata = $( this ).parents('.file-card').data("file");
    file_metadata[$( this ).data("id")] = $( this ).data("value");
    $.post( "/api/updatemetadata", {file_metadata: JSON.stringify(file_metadata), action: "update"} )
    .done(function( metadata ) {
      for (const key of ["selected_metric", "paper_confidence", "usecase_confidence"]) {
          if (metadata[key] != "") {
              $('div[data-id="'+metadata["bucket_file_name"]+'"] .btn-'+key).text(metadata[key]);
          }
      }
    })
    .fail(function() {
      alert( "Something went wrong, please try again." );
    })
});

$(".btn-calculate").on( "click", function(e) {
    e.preventDefault();
    var file_cards = $(".file-card");
    if (file_cards.length < 2) {
        $(".page1-alert").text("Please upload at least two performance tables to create a joint rating.").show();
        return;
    } else {
        $(".page1-alert").hide();
    }

    if ($(".file-card .btn-selected_metric").text().includes("Select")) {
        $(".page1-alert").text("Please select a metric for each performance table.").show();
        return;
    } else {
        $(".page1-alert").hide();
    }

    $(".page").hide();
    $("#page2").show();

    var files = file_cards.map(function() {
        return $(this).data("id");
    }).get();
    $.post( "/api/calculate", {files: JSON.stringify(files), action: "calculate"} )
    .done(function( ranking_table ) {
      $("#result").html(ranking_table);
      $(".page").hide();
      $("#page3").show();
    })
    .fail(function() {
      alert( "Something went wrong, please try again." );
    })
});
