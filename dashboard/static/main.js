fetch("/health")
    .then((res) => res.json())
    .then((data) => {
        document.getElementById("status").textContent = data.status;
    })
    .catch(() => {
        document.getElementById("status").textContent = "unreachable";
    });
