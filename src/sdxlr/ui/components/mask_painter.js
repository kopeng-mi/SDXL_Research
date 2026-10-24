// Mask painter: the image with a paint layer on top. The mask lives in an offscreen canvas at the
// natural size of the image shown (alpha = painted), so strokes are exact however the page is scaled.
// Pointer events cover mouse, touch and pen. Undo keeps the last few snapshots.
// The pixels leave only through export(): a PNG POSTed to the app (the websocket has a 1 MB limit).
const MAX_UNDO = 8;
const AMBER = "rgba(232, 163, 61, 0.55)";

export default {
  template: `
    <div style="position:relative; line-height:0; width:100%;">
      <img ref="img" :src="src" draggable="false" @load="onLoad"
           style="width:100%; height:auto; display:block; max-height:78vh; object-fit:contain; margin:0 auto" />
      <canvas ref="view"
              :style="{position:'absolute', left:0, top:0, width:'100%', height:'100%', touchAction: tool === 'move' ? 'auto' : 'none', cursor: tool === 'move' ? 'default' : 'crosshair'}"
              @pointerdown="down" @pointermove="move" @pointerup="up" @pointercancel="up" @pointerleave="leave"></canvas>
    </div>
  `,
  props: { src: String, tool: String, size: Number, maskurl: String },
  data() {
    return { w: 0, h: 0, drag: null, undo_stack: [], raf: 0 };
  },
  mounted() {
    this.master = document.createElement("canvas");
    // touch screens start in the Move tool, so a finger can still scroll the page
    this.$emit("device", { coarse: window.matchMedia("(pointer: coarse)").matches });
  },
  watch: {
    maskurl() {
      this.loadMask();
    },
    src() {
      this.w = this.h = 0;
    },
  },
  methods: {
    onLoad() {
      const img = this.$refs.img;
      if (!img.naturalWidth) return;
      this.w = this.master.width = this.$refs.view.width = img.naturalWidth;
      this.h = this.master.height = this.$refs.view.height = img.naturalHeight;
      this.undo_stack = [];
      this.loadMask();
      this.render();
      this.$emit("ready", { w: this.w, h: this.h });
    },
    loadMask() {
      if (!this.w) return;
      const ctx = this.master.getContext("2d");
      ctx.clearRect(0, 0, this.w, this.h);
      if (!this.maskurl) {
        this.render();
        this.changed();
        return;
      }
      const m = new Image();
      m.onload = () => {
        // a saved mask of any size is scaled to this canvas; its brightness becomes the paint
        const tmp = document.createElement("canvas");
        tmp.width = this.w;
        tmp.height = this.h;
        const t = tmp.getContext("2d");
        t.drawImage(m, 0, 0, this.w, this.h);
        const d = t.getImageData(0, 0, this.w, this.h);
        for (let i = 0; i < d.data.length; i += 4) {
          d.data[i + 3] = d.data[i];
          d.data[i] = d.data[i + 1] = d.data[i + 2] = 255;
        }
        ctx.putImageData(d, 0, 0);
        this.render();
        this.changed();
      };
      m.src = this.maskurl;
    },
    pos(e) {
      const r = this.$refs.view.getBoundingClientRect();
      return [((e.clientX - r.left) * this.w) / r.width, ((e.clientY - r.top) * this.h) / r.height];
    },
    snapshot() {
      const ctx = this.master.getContext("2d");
      this.undo_stack.push(ctx.getImageData(0, 0, this.w, this.h));
      if (this.undo_stack.length > MAX_UNDO) this.undo_stack.shift();
    },
    down(e) {
      if (this.tool === "move" || !this.w) return;
      e.preventDefault();
      this.$refs.view.setPointerCapture(e.pointerId);
      this.snapshot();
      const [x, y] = this.pos(e);
      this.drag = { x0: x, y0: y, x, y };
      if (this.tool !== "box") this.stroke(x, y, x, y);
      this.schedule();
    },
    move(e) {
      if (!this.drag) return;
      e.preventDefault();
      const [x, y] = this.pos(e);
      if (this.tool !== "box") this.stroke(this.drag.x, this.drag.y, x, y);
      this.drag.x = x;
      this.drag.y = y;
      this.schedule();
    },
    up() {
      if (!this.drag) return;
      const d = this.drag;
      this.drag = null;
      if (this.tool === "box") {
        const ctx = this.master.getContext("2d");
        ctx.fillStyle = "#fff";
        ctx.fillRect(Math.min(d.x0, d.x), Math.min(d.y0, d.y), Math.abs(d.x - d.x0), Math.abs(d.y - d.y0));
      }
      this.render();
      this.changed();
    },
    leave() {},
    stroke(x0, y0, x1, y1) {
      const ctx = this.master.getContext("2d");
      ctx.globalCompositeOperation = this.tool === "eraser" ? "destination-out" : "source-over";
      ctx.strokeStyle = "#fff";
      ctx.lineCap = ctx.lineJoin = "round";
      ctx.lineWidth = this.size || 48;
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1 + 0.01, y1);
      ctx.stroke();
      ctx.globalCompositeOperation = "source-over";
    },
    schedule() {
      if (!this.raf) this.raf = requestAnimationFrame(() => { this.raf = 0; this.render(); });
    },
    render() {
      const v = this.$refs.view;
      if (!v || !this.w) return;
      const ctx = v.getContext("2d");
      ctx.clearRect(0, 0, this.w, this.h);
      ctx.globalCompositeOperation = "source-over";
      ctx.fillStyle = AMBER;
      ctx.fillRect(0, 0, this.w, this.h);
      ctx.globalCompositeOperation = "destination-in"; // the amber tint stays only where the mask is painted
      ctx.drawImage(this.master, 0, 0);
      ctx.globalCompositeOperation = "source-over";
      if (this.drag && this.tool === "box") {
        const d = this.drag;
        ctx.strokeStyle = "#E8A33D";
        ctx.lineWidth = Math.max(2, this.w / 400);
        ctx.strokeRect(Math.min(d.x0, d.x), Math.min(d.y0, d.y), Math.abs(d.x - d.x0), Math.abs(d.y - d.y0));
      }
    },
    isEmpty() {
      const d = this.master.getContext("2d").getImageData(0, 0, this.w, this.h).data;
      for (let i = 3; i < d.length; i += 4) if (d[i] > 8) return false;
      return true;
    },
    changed() {
      this.$emit("changed", { empty: this.w ? this.isEmpty() : true, w: this.w, h: this.h });
    },
    undo() {
      const s = this.undo_stack.pop();
      if (!s) return;
      this.master.getContext("2d").putImageData(s, 0, 0);
      this.render();
      this.changed();
    },
    clear() {
      if (!this.w) return;
      this.snapshot();
      this.master.getContext("2d").clearRect(0, 0, this.w, this.h);
      this.render();
      this.changed();
    },
    invert() {
      if (!this.w) return;
      this.snapshot();
      const tmp = document.createElement("canvas");
      tmp.width = this.w;
      tmp.height = this.h;
      const t = tmp.getContext("2d");
      t.fillStyle = "#fff";
      t.fillRect(0, 0, this.w, this.h);
      t.globalCompositeOperation = "destination-out";
      t.drawImage(this.master, 0, 0);
      const ctx = this.master.getContext("2d");
      ctx.clearRect(0, 0, this.w, this.h);
      ctx.drawImage(tmp, 0, 0);
      this.render();
      this.changed();
    },
    // POST the mask as a PNG (white = repaint, black = keep); resolves to {id} or {error}
    async export(url) {
      if (!this.w) return { error: "The image is not ready yet." };
      const out = document.createElement("canvas");
      out.width = this.w;
      out.height = this.h;
      const o = out.getContext("2d");
      o.fillStyle = "#000";
      o.fillRect(0, 0, this.w, this.h);
      o.drawImage(this.master, 0, 0);
      const blob = await new Promise((res) => out.toBlob(res, "image/png"));
      if (!blob) return { error: "The mask could not be made." };
      try {
        const r = await fetch((window.path_prefix || "") + url, { method: "POST", body: blob });
        if (!r.ok) return { error: "The mask was not accepted (" + r.status + ")." };
        return await r.json();
      } catch (e) {
        return { error: "The mask could not be sent. Check the connection and try again." };
      }
    },
  },
};
