// Copyright (c) ONNX Project Contributors
// SPDX-License-Identifier: Apache-2.0

//! Doubly-linked list for safe graph mutation.
//!
//! This module provides a doubly-linked list that supports safe iteration
//! while the list is being mutated. This is essential for graph transformation
//! passes that need to iterate over nodes while adding or removing them.

use std::cell::{Cell, RefCell};
use std::fmt;
use std::marker::PhantomData;
use std::rc::Rc;

/// A link in a doubly linked list.
///
/// The link box is a container for the actual value in the list. It maintains
/// the links between elements in the linked list. The actual value is stored
/// in the `value` attribute.
///
/// By using a separate container for the actual value, we can safely remove
/// the value from the list without losing the links. This allows us to remove
/// values during iteration and place them into a different list without
/// breaking any chains.
struct LinkBox<T> {
    prev: Cell<Option<Rc<LinkBox<T>>>>,
    next: Cell<Option<Rc<LinkBox<T>>>>,
    value: RefCell<Option<T>>,
}

impl<T> LinkBox<T> {
    /// Creates a new link box with a value.
    fn new(value: T) -> Rc<Self> {
        Rc::new(Self {
            prev: Cell::new(None),
            next: Cell::new(None),
            value: RefCell::new(Some(value)),
        })
    }

    /// Creates a sentinel (root) link box without a value.
    fn sentinel() -> Rc<Self> {
        Rc::new(Self {
            prev: Cell::new(None),
            next: Cell::new(None),
            value: RefCell::new(None),
        })
    }

    /// Checks if this link has been erased (value removed).
    fn is_erased(&self) -> bool {
        self.value.borrow().is_none()
    }

    /// Erases the value from this link, removing it from the list.
    fn erase(&self) -> Option<T> {
        if self.is_erased() {
            return None;
        }

        // Get the previous and next links without removing them from self yet
        let prev = self.prev.replace(None);
        let next = self.next.replace(None);

        // Update the links to bypass this node
        if let (Some(prev_link), Some(next_link)) = (&prev, &next) {
            prev_link.next.set(Some(next_link.clone()));
            next_link.prev.set(Some(prev_link.clone()));
        }

        // Restore the links on self (though they won't be used)
        self.prev.set(prev);
        self.next.set(next);

        // Remove and return the value
        self.value.borrow_mut().take()
    }
}

impl<T: fmt::Debug> fmt::Debug for LinkBox<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LinkBox")
            .field("value", &self.value.borrow())
            .field("erased", &self.is_erased())
            .finish()
    }
}

/// A doubly-linked ordered list that supports safe mutation during iteration.
///
/// The container maintains the order of elements and allows safe iteration
/// while the list is being mutated. Adding and removing elements from the
/// list during iteration is safe.
///
/// During iteration:
/// - If new elements are inserted after the current node, the iterator will
///   iterate over them as well.
/// - If new elements are inserted before the current node, they will not be
///   iterated over in this iteration.
/// - If the current node is removed and inserted in a different location,
///   iteration will start from the "next" node at the original location.
///
/// Time complexity:
/// - Inserting and removing nodes: O(1)
/// - Accessing nodes by index: O(n)
/// - Accessing nodes at either end: O(1)
///
/// # Examples
///
/// ```
/// use onnx_ir_core::linked_list::DoublyLinkedList;
///
/// let mut list = DoublyLinkedList::new();
/// list.push_back(1);
/// list.push_back(2);
/// list.push_back(3);
///
/// assert_eq!(list.len(), 3);
/// assert_eq!(list.front(), Some(&1));
/// assert_eq!(list.back(), Some(&3));
/// ```
pub struct DoublyLinkedList<T> {
    root: Rc<LinkBox<T>>,
    length: usize,
}

impl<T> DoublyLinkedList<T> {
    /// Creates a new empty list.
    pub fn new() -> Self {
        let root = LinkBox::sentinel();
        root.prev.set(Some(root.clone()));
        root.next.set(Some(root.clone()));
        
        Self {
            root,
            length: 0,
        }
    }

    /// Returns the number of elements in the list.
    pub fn len(&self) -> usize {
        self.length
    }

    /// Returns true if the list is empty.
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Appends an element to the back of the list.
    pub fn push_back(&mut self, value: T) {
        let new_link = LinkBox::new(value);
        self.insert_before_link(self.root.clone(), new_link);
    }

    /// Appends an element to the front of the list.
    pub fn push_front(&mut self, value: T) {
        let new_link = LinkBox::new(value);
        let first = self.root.next.take().unwrap();
        self.root.next.set(Some(first.clone()));
        self.insert_before_link(first, new_link);
    }

    /// Removes and returns the element at the back of the list.
    pub fn pop_back(&mut self) -> Option<T> {
        if self.is_empty() {
            return None;
        }

        let last = self.root.prev.replace(None).unwrap();
        self.root.prev.set(Some(last.clone()));
        
        if !last.is_erased() && !Rc::ptr_eq(&last, &self.root) {
            self.length -= 1;
            return last.erase();
        }
        
        None
    }

    /// Removes and returns the element at the front of the list.
    pub fn pop_front(&mut self) -> Option<T> {
        if self.is_empty() {
            return None;
        }

        let first = self.root.next.replace(None).unwrap();
        self.root.next.set(Some(first.clone()));
        
        if !first.is_erased() && !Rc::ptr_eq(&first, &self.root) {
            self.length -= 1;
            return first.erase();
        }
        
        None
    }

    /// Returns a reference to the front element.
    pub fn front(&self) -> Option<&T> {
        if self.is_empty() {
            return None;
        }

        let first = self.root.next.replace(None)?;
        self.root.next.set(Some(first.clone()));
        
        unsafe {
            // SAFETY: We know the value exists if the list is not empty
            // and we're returning a reference with the same lifetime as self
            (*first.value.as_ptr()).as_ref()
        }
    }

    /// Returns a reference to the back element.
    pub fn back(&self) -> Option<&T> {
        if self.is_empty() {
            return None;
        }

        let last = self.root.prev.replace(None)?;
        self.root.prev.set(Some(last.clone()));
        
        unsafe {
            // SAFETY: We know the value exists if the list is not empty
            // and we're returning a reference with the same lifetime as self
            (*last.value.as_ptr()).as_ref()
        }
    }

    /// Clears the list, removing all elements.
    pub fn clear(&mut self) {
        while self.pop_front().is_some() {}
    }

    /// Returns an iterator over the list.
    pub fn iter(&self) -> Iter<'_, T> {
        let next = self.root.next.replace(None);
        self.root.next.set(next.clone());
        
        Iter {
            current: next,
            root: self.root.clone(),
            _marker: PhantomData,
        }
    }

    /// Helper to insert a link before another link.
    fn insert_before_link(&mut self, before: Rc<LinkBox<T>>, new_link: Rc<LinkBox<T>>) {
        let prev = before.prev.take().unwrap();
        before.prev.set(Some(prev.clone()));

        new_link.prev.set(Some(prev.clone()));
        new_link.next.set(Some(before.clone()));
        prev.next.set(Some(new_link.clone()));
        before.prev.set(Some(new_link));

        self.length += 1;
    }
}

impl<T> Default for DoublyLinkedList<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: fmt::Debug> fmt::Debug for DoublyLinkedList<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

/// An iterator over the elements of a `DoublyLinkedList`.
pub struct Iter<'a, T> {
    current: Option<Rc<LinkBox<T>>>,
    root: Rc<LinkBox<T>>,
    _marker: PhantomData<&'a T>,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let current = self.current.clone()?;
            
            // Check if we've circled back to the root
            if Rc::ptr_eq(&current, &self.root) {
                return None;
            }

            // Move to next
            let next = current.next.replace(None);
            current.next.set(next.clone());
            self.current = next;

            // Return the value if not erased
            if !current.is_erased() {
                unsafe {
                    // SAFETY: We're creating a reference with lifetime 'a
                    // The borrow checker ensures this is safe
                    return (*current.value.as_ptr()).as_ref();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_doubly_linked_list_basic() {
        let mut list = DoublyLinkedList::new();
        
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
        
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        
        assert_eq!(list.len(), 3);
        assert_eq!(list.front(), Some(&1));
        assert_eq!(list.back(), Some(&3));
    }

    #[test]
    fn test_doubly_linked_list_push_front() {
        let mut list = DoublyLinkedList::new();
        
        list.push_front(1);
        list.push_front(2);
        list.push_front(3);
        
        assert_eq!(list.len(), 3);
        assert_eq!(list.front(), Some(&3));
        assert_eq!(list.back(), Some(&1));
    }

    #[test]
    fn test_doubly_linked_list_pop() {
        let mut list = DoublyLinkedList::new();
        
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        
        assert_eq!(list.pop_front(), Some(1));
        assert_eq!(list.pop_back(), Some(3));
        assert_eq!(list.pop_front(), Some(2));
        assert_eq!(list.pop_front(), None);
        
        assert!(list.is_empty());
    }

    #[test]
    fn test_doubly_linked_list_iter() {
        let mut list = DoublyLinkedList::new();
        
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        
        let collected: Vec<_> = list.iter().copied().collect();
        assert_eq!(collected, vec![1, 2, 3]);
    }

    #[test]
    fn test_doubly_linked_list_clear() {
        let mut list = DoublyLinkedList::new();
        
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        
        list.clear();
        
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
    }
}
