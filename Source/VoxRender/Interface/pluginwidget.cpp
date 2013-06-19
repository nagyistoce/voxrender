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

// Include Headers
#include "ui_pluginwidget.h"
#include "pluginwidget.h"

// Include Dependencies
#include "VoxLib/Core/Format.h"
#include "VoxLib/Plugin/PluginManager.h"

#include <boost/filesystem.hpp>

using namespace vox;

// ----------------------------------------------------------------------------
//  Constuctor
// ----------------------------------------------------------------------------
PluginWidget::PluginWidget(QWidget *parent, std::shared_ptr<PluginInfo> info) : 
	QWidget(parent), 
    ui(new Ui::PluginWidget)
{
	ui->setupUi(this);

    m_info = info;

    ui->lineEdit_version->setText( QString::fromLatin1(
        format("%1%.%2%.%3%", m_info->version.major, m_info->version.minor, m_info->version.patch).c_str()
        ) );

    boost::filesystem::path filePath(m_info->file);

    String fileLink = format("<a href='%1%'>%2%</a>", "file://" + filePath.remove_filename().string(), m_info->file);

    ui->lineEdit_name->setText( QString::fromLatin1(m_info->name.c_str()) );
    ui->lineEdit_vendor->setText( QString::fromLatin1(m_info->vendor.c_str()) );
    ui->lineEdit_url->setText( QString::fromLatin1(m_info->url.c_str()) );
    ui->label_file->setText( QString::fromLatin1(fileLink.c_str()) );
    ui->textBrowser_desc->setText( QString::fromLatin1(m_info->description.c_str()) );
    ui->label_file->setOpenExternalLinks(true);

    auto & pm = PluginManager::instance();

    this->blockSignals(true);
    setEnabled( pm.isLoaded(info) );
    this->blockSignals(false);
    // :TODO: Connect on/off signals from parent pane to plugin manager's enable/disable methods and init state
}
    
// ----------------------------------------------------------------------------
//  Clear UI
// ----------------------------------------------------------------------------
PluginWidget::~PluginWidget()
{
    delete ui;
}

// ----------------------------------------------------------------------------
//  Enables or disables the specified plugin using the VoxLib manager
// ----------------------------------------------------------------------------
void PluginWidget::changeEvent(QEvent * e)
{
    if (e->type() == QEvent::EnabledChange)
    {
        auto & pm = PluginManager::instance();

        if (isEnabled())
        {
            pm.load(m_info);
        }
        else
        {
            pm.unload(m_info);
        }
    }
}