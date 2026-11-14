//! Which task is changing which file.
//!
//! Tasks run side by side in the same folders. So that two of them do not
//! write over each other, the first task to change a file holds it until
//! that task ends; another task's change to the same file is refused, and
//! the model is told who holds it and to come back to it later. Nothing
//! here waits, so two tasks can never block each other for good.
//!
//! This covers the file tools (`write_file`, `edit_file`). What a shell
//! command writes cannot be known in advance and is not covered.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Default)]
pub struct Claims {
    held: HashMap<PathBuf, u32>,
}

impl Claims {
    /// Takes `path` for `task`, or says which other task holds it.
    pub fn claim(&mut self, path: &Path, task: u32) -> Result<(), u32> {
        match self.held.get(path) {
            Some(holder) if *holder != task => Err(*holder),
            _ => {
                self.held.insert(path.to_path_buf(), task);
                Ok(())
            }
        }
    }

    /// Lets go of everything `task` holds. Returns how many files that was.
    pub fn release(&mut self, task: u32) -> usize {
        let before = self.held.len();
        self.held.retain(|_, holder| *holder != task);
        before - self.held.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_task_to_change_a_file_holds_it_until_it_ends() {
        let mut claims = Claims::default();
        let (a, b) = (Path::new("/p/a.txt"), Path::new("/p/b.txt"));
        assert_eq!(claims.claim(a, 1), Ok(()));
        // The holder may change it again; another task may not.
        assert_eq!(claims.claim(a, 1), Ok(()));
        assert_eq!(claims.claim(a, 2), Err(1));
        // Other files are free.
        assert_eq!(claims.claim(b, 2), Ok(()));
        assert_eq!(claims.claim(b, 1), Err(2));
        // When a task ends, what it held is free again.
        assert_eq!(claims.release(1), 1);
        assert_eq!(claims.claim(a, 2), Ok(()));
        assert_eq!(claims.release(2), 2);
        assert_eq!(claims.release(2), 0);
    }
}
