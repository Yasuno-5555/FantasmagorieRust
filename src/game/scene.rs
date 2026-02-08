use crate::game::input::ActionState;
use crate::draw::DrawList;

pub enum SceneTransition {
    None,
    Push(Box<dyn Scene>),
    Pop,
    Replace(Box<dyn Scene>),
    Quit,
}

pub trait Scene {
    fn on_enter(&mut self) {}
    fn on_exit(&mut self) {}
    fn update(&mut self, dt: f32, input: &ActionState) -> SceneTransition;
    fn draw(&mut self, dl: &mut DrawList);
}

pub struct SceneManager {
    stack: Vec<Box<dyn Scene>>,
}

impl SceneManager {
    pub fn new(initial_scene: Box<dyn Scene>) -> Self {
        let mut sm = Self {
            stack: vec![initial_scene],
        };
        if let Some(s) = sm.stack.last_mut() {
            s.on_enter();
        }
        sm
    }

    pub fn update(&mut self, dt: f32, input: &ActionState) -> bool {
        if self.stack.is_empty() {
            return false;
        }

        let transition = {
            let scene = self.stack.last_mut().unwrap();
            scene.update(dt, input)
        };

        match transition {
            SceneTransition::None => {},
            SceneTransition::Push(mut scene) => {
                if let Some(curr) = self.stack.last_mut() {
                    curr.on_exit();
                }
                scene.on_enter();
                self.stack.push(scene);
            },
            SceneTransition::Pop => {
                if let Some(mut curr) = self.stack.pop() {
                    curr.on_exit();
                }
                if let Some(next) = self.stack.last_mut() {
                    next.on_enter();
                }
            },
            SceneTransition::Replace(mut scene) => {
                if let Some(mut curr) = self.stack.pop() {
                    curr.on_exit();
                }
                scene.on_enter();
                self.stack.push(scene);
            },
            SceneTransition::Quit => {
                while let Some(mut curr) = self.stack.pop() {
                    curr.on_exit();
                }
                return false;
            }
        }

        !self.stack.is_empty()
    }

    pub fn draw(&mut self, dl: &mut DrawList) {
        // Draw only top scene? Or draw all (overlay)?
        // Stack usually implies overlay.
        // Let's draw all from bottom to top.
        for scene in &mut self.stack {
            scene.draw(dl);
        }
    }
}
