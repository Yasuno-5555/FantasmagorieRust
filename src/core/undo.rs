//! Undo/Redo System
//!
//! Provides command pattern based undo/redo with:
//! - Command trait for actions
//! - CommandStack for history management
//! - Value change commands
//! - Batch/group commands

use std::collections::VecDeque;

/// Command trait for undoable actions
pub trait Command: Send {
    /// Execute the command
    fn execute(&mut self);
    
    /// Undo the command
    fn undo(&mut self);
    
    /// Get command description
    fn description(&self) -> &str;
    
    /// Check if command can be merged with another
    fn can_merge(&self, _other: &dyn Command) -> bool {
        false
    }
    
    /// Merge with another command (for continuous edits like typing)
    fn merge(&mut self, _other: Box<dyn Command>) -> bool {
        false
    }
}

/// Value change command for simple property changes
pub struct ValueChangeCommand<T: Clone + PartialEq + Send + 'static> {
    target: *mut T,
    old_value: T,
    new_value: T,
    description: String,
}

// SAFETY: The target pointer is only accessed when executing/undoing
unsafe impl<T: Clone + PartialEq + Send + 'static> Send for ValueChangeCommand<T> {}

impl<T: Clone + PartialEq + Send + 'static> ValueChangeCommand<T> {
    /// Create a new value change command
    /// 
    /// # Safety
    /// The target pointer must remain valid for the lifetime of this command
    pub unsafe fn new(target: &mut T, new_value: T, description: &str) -> Self {
        Self {
            target: target as *mut T,
            old_value: target.clone(),
            new_value,
            description: description.to_string(),
        }
    }
}

impl<T: Clone + PartialEq + Send + 'static> Command for ValueChangeCommand<T> {
    fn execute(&mut self) {
        unsafe {
            *self.target = self.new_value.clone();
        }
    }

    fn undo(&mut self) {
        unsafe {
            *self.target = self.old_value.clone();
        }
    }

    fn description(&self) -> &str {
        &self.description
    }
}

/// Callback-based command for complex actions
pub struct CallbackCommand {
    execute_fn: Box<dyn FnMut() + Send>,
    undo_fn: Box<dyn FnMut() + Send>,
    description: String,
}

impl CallbackCommand {
    pub fn new<E, U>(execute: E, undo: U, description: &str) -> Self
    where
        E: FnMut() + Send + 'static,
        U: FnMut() + Send + 'static,
    {
        Self {
            execute_fn: Box::new(execute),
            undo_fn: Box::new(undo),
            description: description.to_string(),
        }
    }
}

impl Command for CallbackCommand {
    fn execute(&mut self) {
        (self.execute_fn)();
    }

    fn undo(&mut self) {
        (self.undo_fn)();
    }

    fn description(&self) -> &str {
        &self.description
    }
}

/// Batch command for grouping multiple commands
pub struct BatchCommand {
    commands: Vec<Box<dyn Command>>,
    description: String,
}

impl BatchCommand {
    pub fn new(description: &str) -> Self {
        Self {
            commands: Vec::new(),
            description: description.to_string(),
        }
    }

    pub fn add<C: Command + 'static>(&mut self, command: C) {
        self.commands.push(Box::new(command));
    }

    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }
}

impl Command for BatchCommand {
    fn execute(&mut self) {
        for cmd in &mut self.commands {
            cmd.execute();
        }
    }

    fn undo(&mut self) {
        // Undo in reverse order
        for cmd in self.commands.iter_mut().rev() {
            cmd.undo();
        }
    }

    fn description(&self) -> &str {
        &self.description
    }
}

/// Command stack for managing undo/redo history
pub struct CommandStack {
    undo_stack: VecDeque<Box<dyn Command>>,
    redo_stack: Vec<Box<dyn Command>>,
    max_size: usize,
    /// Is currently in batch mode
    batch: Option<BatchCommand>,
}

impl CommandStack {
    /// Create a new command stack
    pub fn new() -> Self {
        Self::with_max_size(100)
    }

    /// Create with custom max size
    pub fn with_max_size(max_size: usize) -> Self {
        Self {
            undo_stack: VecDeque::new(),
            redo_stack: Vec::new(),
            max_size,
            batch: None,
        }
    }

    /// Push and execute a command
    pub fn push<C: Command + 'static>(&mut self, mut command: C) {
        // If in batch mode, add to batch
        if let Some(ref mut batch) = self.batch {
            command.execute();
            batch.add(command);
            return;
        }

        // Clear redo stack when new command is pushed
        self.redo_stack.clear();

        // Execute the command
        command.execute();

        // Add to undo stack (merging disabled for simplicity)
        self.undo_stack.push_back(Box::new(command));

        // Limit stack size
        while self.undo_stack.len() > self.max_size {
            self.undo_stack.pop_front();
        }
    }

    /// Start a batch of commands
    pub fn begin_batch(&mut self, description: &str) {
        self.batch = Some(BatchCommand::new(description));
    }

    /// End the current batch
    pub fn end_batch(&mut self) {
        if let Some(batch) = self.batch.take() {
            if !batch.is_empty() {
                self.redo_stack.clear();
                self.undo_stack.push_back(Box::new(batch));
                while self.undo_stack.len() > self.max_size {
                    self.undo_stack.pop_front();
                }
            }
        }
    }

    /// Cancel the current batch
    pub fn cancel_batch(&mut self) {
        if let Some(mut batch) = self.batch.take() {
            // Undo all commands in the batch
            batch.undo();
        }
    }

    /// Check if undo is available
    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    /// Check if redo is available
    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    /// Undo the last command
    pub fn undo(&mut self) -> Option<&str> {
        if let Some(mut cmd) = self.undo_stack.pop_back() {
            cmd.undo();
            let desc = cmd.description().to_string();
            self.redo_stack.push(cmd);
            Some(self.redo_stack.last().map(|c| c.description()).unwrap_or(""))
        } else {
            None
        }
    }

    /// Redo the last undone command
    pub fn redo(&mut self) -> Option<&str> {
        if let Some(mut cmd) = self.redo_stack.pop() {
            cmd.execute();
            let desc = cmd.description().to_string();
            self.undo_stack.push_back(cmd);
            Some(self.undo_stack.back().map(|c| c.description()).unwrap_or(""))
        } else {
            None
        }
    }

    /// Clear all history
    pub fn clear(&mut self) {
        self.undo_stack.clear();
        self.redo_stack.clear();
        self.batch = None;
    }

    /// Get undo count
    pub fn undo_count(&self) -> usize {
        self.undo_stack.len()
    }

    /// Get redo count
    pub fn redo_count(&self) -> usize {
        self.redo_stack.len()
    }

    /// Get description of next undo action
    pub fn undo_description(&self) -> Option<&str> {
        self.undo_stack.back().map(|c| c.description())
    }

    /// Get description of next redo action
    pub fn redo_description(&self) -> Option<&str> {
        self.redo_stack.last().map(|c| c.description())
    }
}

impl Default for CommandStack {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn test_callback_command() {
        let value = Arc::new(Mutex::new(0));
        let v1 = value.clone();
        let v2 = value.clone();

        let mut cmd = CallbackCommand::new(
            move || *v1.lock().unwrap() = 10,
            move || *v2.lock().unwrap() = 0,
            "Set value to 10"
        );

        cmd.execute();
        assert_eq!(*value.lock().unwrap(), 10);

        cmd.undo();
        assert_eq!(*value.lock().unwrap(), 0);
    }

    #[test]
    fn test_command_stack() {
        let value = Arc::new(Mutex::new(0));
        let mut stack = CommandStack::new();

        let v1 = value.clone();
        let v2 = value.clone();
        stack.push(CallbackCommand::new(
            move || *v1.lock().unwrap() = 10,
            move || *v2.lock().unwrap() = 0,
            "Set to 10"
        ));
        assert_eq!(*value.lock().unwrap(), 10);

        stack.undo();
        assert_eq!(*value.lock().unwrap(), 0);

        stack.redo();
        assert_eq!(*value.lock().unwrap(), 10);
    }

    #[test]
    fn test_batch_command() {
        let value = Arc::new(Mutex::new(0));
        let mut stack = CommandStack::new();

        stack.begin_batch("Increment twice");

        let v1 = value.clone();
        let v2 = value.clone();
        stack.push(CallbackCommand::new(
            move || *v1.lock().unwrap() += 5,
            move || *v2.lock().unwrap() -= 5,
            "Add 5"
        ));

        let v3 = value.clone();
        let v4 = value.clone();
        stack.push(CallbackCommand::new(
            move || *v3.lock().unwrap() += 7,
            move || *v4.lock().unwrap() -= 7,
            "Add 7"
        ));

        stack.end_batch();
        assert_eq!(*value.lock().unwrap(), 12);
        assert_eq!(stack.undo_count(), 1); // Single batch command

        stack.undo();
        assert_eq!(*value.lock().unwrap(), 0);
    }
}
