document.addEventListener("DOMContentLoaded", () => {
    const filterRoots = document.querySelectorAll("[data-filter-root]");

    filterRoots.forEach((root) => {
        const input = root.querySelector("[data-filter-input]");
        const items = root.querySelectorAll("[data-filter-item]");
        const emptyState = root.querySelector("[data-filter-empty]");

        if (!input || !items.length) {
            return;
        }

        const applyFilter = () => {
            const query = input.value.trim().toLowerCase();
            let visibleCount = 0;

            items.forEach((item) => {
                const text = item.dataset.filterText || "";
                const matches = !query || text.includes(query);
                item.classList.toggle("hidden", !matches);
                if (matches) {
                    visibleCount += 1;
                }
            });

            if (emptyState) {
                emptyState.classList.toggle("hidden", visibleCount !== 0);
            }
        };

        input.addEventListener("input", applyFilter);
    });
});
