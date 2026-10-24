// Before/after slider: two stacked images, the top one clipped at the handle. Pure client side, so
// dragging never waits for the server. Pointer events work for mouse, touch and pen; vertical swipes
// still scroll the page (touch-action: pan-y).
export default {
  template: `
    <div ref="box" style="position:relative; line-height:0; width:100%; user-select:none; touch-action:pan-y; cursor:ew-resize"
         @pointerdown="down" @pointermove="move" @pointerup="up" @pointercancel="up">
      <img :src="after" draggable="false" style="width:100%; height:auto; display:block; max-height:78vh; object-fit:contain; margin:0 auto" />
      <img :src="before" draggable="false"
           :style="{position:'absolute', left:0, top:0, width:'100%', height:'100%', objectFit:'contain', clipPath:'inset(0 ' + (100 - pos) + '% 0 0)'}" />
      <div :style="{position:'absolute', top:0, bottom:0, left:pos + '%', width:'2px', background:'#fff', boxShadow:'0 0 4px #000'}">
        <div style="position:absolute; top:50%; left:-21px; width:44px; height:44px; margin-top:-22px; border-radius:22px; background:rgba(0,0,0,.55); border:2px solid #fff; color:#fff; font-size:20px; line-height:40px; text-align:center">&#8596;</div>
      </div>
      <div style="position:absolute; left:8px; top:8px; padding:2px 8px; border-radius:6px; background:rgba(0,0,0,.55); color:#fff; font-size:12px; line-height:18px">Before</div>
      <div style="position:absolute; right:8px; top:8px; padding:2px 8px; border-radius:6px; background:rgba(0,0,0,.55); color:#fff; font-size:12px; line-height:18px">After</div>
    </div>
  `,
  props: { before: String, after: String },
  data() {
    return { pos: 50, dragging: false };
  },
  methods: {
    set(e) {
      const r = this.$refs.box.getBoundingClientRect();
      this.pos = Math.max(0, Math.min(100, ((e.clientX - r.left) / r.width) * 100));
    },
    down(e) {
      this.dragging = true;
      this.$refs.box.setPointerCapture(e.pointerId);
      this.set(e);
    },
    move(e) {
      if (this.dragging) this.set(e);
    },
    up() {
      this.dragging = false;
    },
  },
};
