document.getElementById('dropTarget').addEventListener('dragenter', event => {
    event.preventDefault();
    event.target.classList.add('dragging');
});

document.getElementById('dropTarget').addEventListener('dragleave', event => {
    event.preventDefault();
    event.target.classList.remove('dragging');
});

document.getElementById('dropTarget').addEventListener('dragover', event => {
    event.preventDefault();
})

document.getElementById('dropTarget').addEventListener('drop', async event => {
    event.preventDefault();
    event.target.classList.remove('dragging');

    if (event.dataTransfer.items.length !== 1 || event.dataTransfer.items[0].kind !== 'file') {
        return;
    }

    const photo = event.dataTransfer.items[0].getAsFile();

    const data = new FormData();
    data.append('photo', photo)

    response = await fetch('/submit', {
        method: 'POST',
        body: data,
    });

    if (!response.ok) {
        document.getElementById('responseText').innerText = await response.text();
        document.getElementById('uploadedImage').classList.remove('uploaded');
        return;
    }

    document.getElementById('uploadedImage').src = URL.createObjectURL(photo);
    document.getElementById('uploadedImage').classList.add('uploaded');
    document.getElementById('responseText').innerText = await response.text();
});