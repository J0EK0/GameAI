class MLPlay:
    def __init__(self, ai_name, *args, **kwargs):
        self.player = ai_name
        self.target_x = 1240
        self.target_y = 770

    def update(self, scene_info: dict, *args, **kwargs):
        if isinstance(scene_info, list):
            scene_info = scene_info[0]
        if scene_info["status"] != "GAME_ALIVE":
            return []

        sx, sy = scene_info["self_x"], scene_info["self_y"]

        # 一直往右下移動直到抵達邊界
        if sx < self.target_x and sy < self.target_y:
            return ["DOWN"] if (self.target_y - sy) > (self.target_x - sx) else ["RIGHT"]
        elif sx < self.target_x:
            return ["RIGHT"]
        elif sy < self.target_y:
            return ["DOWN"]
        else:
            return ["RIGHT"]  # 已在邊界，保持其中一個方向

    def reset(self):
        pass
