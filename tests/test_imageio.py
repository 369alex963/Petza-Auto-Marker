from PIL import Image

from petza.imageio import list_images, load_image


def test_exif_rotation_applied(tmp_path):
    img = Image.new("RGB", (40, 20), "red")
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW on display
    p = tmp_path / "r.jpg"
    img.save(p, exif=exif)
    assert load_image(p).shape == (40, 20, 3)


def test_list_images(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / ".hidden").mkdir()
    for name in ("b.jpg", "sub/a.PNG", ".hidden/x.jpg", "notes.txt"):
        Image.new("RGB", (4, 4)).save(tmp_path / name, format="PNG") if not name.endswith(".txt") else (tmp_path / name).write_text("x")
    names = [p.name for p in list_images(tmp_path)]
    assert names == ["b.jpg", "a.PNG"]
