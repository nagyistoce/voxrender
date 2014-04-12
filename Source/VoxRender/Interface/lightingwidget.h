/* ===========================================================================

	Project: VoxRender

	Description: Implements a control widget for editing the light set

    Copyright (C) 2014 Lucas Sherman

	Lucas Sherman, email: LucasASherman@gmail.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

=========================================================================== */

// Begin Definition
#ifndef LIGHTNIG_WIDGET_H
#define LIGHTNIG_WIDGET_H

// QT Dependencies
#include <QtWidgets/QWidget>
#include <QtWidgets/QLayoutItem>

// Include Dependencies
#include "VoxScene/Light.h"
#include "panewidget.h"

// Point light interface
class LightingWidget : public QWidget
{
	Q_OBJECT

public:
    /** Constructor */
	explicit LightingWidget(QWidget * parent, QLayout * layout);

    /** Destructor */
	~LightingWidget();

private:
    /** Adds a light to the scene */
    void add(std::shared_ptr<vox::Light> light);

    /** Removes a light from the scene */
    void remove(std::shared_ptr<vox::Light> light);

    PaneWidget *           m_ambientPane;   ///< Ambient light widget
	std::list<PaneWidget*> m_panes;    ///< Other light widgets
    QSpacerItem *          m_spacer;        ///< Spacing element for pane list
    QLayout *              m_layout;        ///< Layout for light panes
    QWidget *              m_parent;        ///< Parent widget 

private slots:
    /** Slot for scene change events */
    void sceneChanged();

    /** Pane removal slot called on light deletion */
    void remove(PaneWidget * pane);
};

#endif // LIGHTNIG_WIDGET_H

