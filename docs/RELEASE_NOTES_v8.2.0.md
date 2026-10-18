# Fizgig v8.2.0: Studio

![Studio with Keep in view pinned to the top and I'm feeling lucky's panel](https://raw.githubusercontent.com/shootthesound/Fizgig/master/docs/assets/studio_keep_in_view.png)

**I'm feeling lucky has its own panel, and LoRA the Explorer's way of working is back in the Studio:** roll, keep what you like, roll the rest.

![I'm feeling lucky](https://raw.githubusercontent.com/shootthesound/Fizgig/master/docs/assets/studio_lucky_pill.png)

- **Roll** sets how many blocks a roll changes: 1, 2, 4 and up to the model's block count, or All.
- **Freeze tweaked** locks every block you've moved, by hand or by a roll, so the next rolls leave them alone. Press it again after a roll to freeze the new ones too. **Unfreeze** lets them go; blocks you locked yourself stay locked.
- The panel shows how many sliders are still free to roll.
- With a donor loaded, pick whether a roll changes the primary, the donor or both.
- **Slider strengths from … to …** sets the range a roll picks from.

**Keep in view:** the Preview card's button puts Baseline and Tweaked in a panel that stays on screen while you work on the sliders. Float it anywhere and resize it, or pin it to the top of the tab and drag its edge to set its height. Pictures and clips scale to fit, and it remembers how you left it.

**Donor LoRAs:**
- The left pane is now your LoRA on its own, with the donor never in it, for pictures, clips and songs. Its caption says so.
- A donor starts fully on in its blocks.

**Faster first preview:** with no slider moved, one render fills both panes.

**Fixes:**
- The taskbar shows Fizgig's icon again.
- The picture viewer no longer stops on a blank No LoRA picture.
- Switching between picture, video and music models clears the last model's previews.
- The buttons, dropdowns and boxes above the sliders line up at one height.
