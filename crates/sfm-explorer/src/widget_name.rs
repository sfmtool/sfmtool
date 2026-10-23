// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! A name for a control that no visible text names.

/// Give the widget of `response` `name` as its own AccessKit label, which is
/// the name a screen reader reads and `get_widgets` lists it under.
///
/// For a control that no text beside it names: a number box whose unit is a
/// suffix inside its value, or one followed by a unit alone. A control with a
/// label beside it is linked to that label with `Response::labelled_by`
/// instead. A number box keeps the name while it is open for typing, since
/// egui gives its text field the box's id.
pub(crate) fn set_widget_name(response: &egui::Response, name: &str) {
    response
        .ctx
        .accesskit_node_builder(response.id, |node| node.set_label(name));
}
