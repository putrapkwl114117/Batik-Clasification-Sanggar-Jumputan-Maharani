document.getElementById("toggleSidebar").addEventListener("click", function () {
  document.getElementById("sidebar").classList.toggle("collapsed");
  document.querySelector(".content").classList.toggle("collapsed");
});

document.getElementById("addProductBtn").addEventListener("click", function () {
  const modal = new bootstrap.Modal(document.getElementById("addProductModal"));
  modal.show();
});

document.getElementById("productBtn").addEventListener("click", function () {
  // Logic to load product content or redirect to product page
  alert("Product section clicked!");
});
