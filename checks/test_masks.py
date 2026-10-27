"""Masked training's folder check (the Training tab's note, and Start's "no masks" refusal): a photo stored sideways
with a rotation flag (EXIF) trains upright, so its mask is checked against the upright size."""
import os

from PIL import Image

from fizgig.dataset import masks


def _rotated_photo(folder, mask_size):
    pic = Image.new("RGB", (600, 400), "gray")       # stored 600x400, seen 400x600 (flag 6: turn a quarter)
    exif = pic.getexif()
    exif[0x0112] = 6
    pic.save(folder / "photo.jpg", exif=exif)
    Image.new("L", mask_size, 255).save(folder / "photo-masklabel.png")


def test_upright_mask_on_a_rotated_photo_counts(tmp_path):
    _rotated_photo(tmp_path, (400, 600))
    rep = masks.folder_report([str(tmp_path)], (".jpg",))
    assert (rep["with"], rep["total"], rep["bad"]) == (1, 1, [])


def test_sideways_mask_on_a_rotated_photo_is_named(tmp_path):
    _rotated_photo(tmp_path, (600, 400))
    rep = masks.folder_report([str(tmp_path)], (".jpg",))
    assert rep["with"] == 0 and len(rep["bad"]) == 1 and "shape of its picture" in rep["bad"][0]


def test_look_filter_leaves_masks_out(tmp_path):
    # the Look Consistency Filter scores the pictures, not the masks made beside them (#247)
    from fizgig.dataset import prep
    for n in ("a.png", "a-masklabel.png", "b.jpg", "b-masklabel.png"):
        Image.new("RGB", (8, 8)).save(tmp_path / n)
    assert [os.path.basename(f) for f in prep.look_files(str(tmp_path))] == ["a.png", "b.jpg"]
