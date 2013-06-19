/* ===========================================================================

	Project: VoxRender

	Description: Pane widget which depicts plugin information for user

    Copyright (C) 2013 Lucas Sherman

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
#ifndef PLUGIN_WIDGET_H
#define PLUGIN_WIDGET_H

// Include Dependencies
#include "VoxLib/Plugin/PluginInfo.h"

#include <QtWidgets/QWidget>

namespace Ui {
	class PluginWidget;
}

// Volume data histogram widget
class PluginWidget : public QWidget
{
	Q_OBJECT

public:
    /** Creates a new plugin info display widget for the specified plugin info */
	explicit PluginWidget(QWidget *parent, std::shared_ptr<vox::PluginInfo> info);

	~PluginWidget();

protected:
    /** Enables or disables the plugin  */
	virtual void changeEvent(QEvent *e);

private:
	Ui::PluginWidget * ui;

    std::shared_ptr<vox::PluginInfo> m_info;
};

// End definition
#endif // PLUGIN_WIDGET_H

