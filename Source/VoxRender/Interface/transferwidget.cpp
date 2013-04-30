/* ===========================================================================

	Project: TransferWidget - Transfer Function Widget

	Description:
	 Implements an interface for interactive transfer function modification.

    Copyright (C) 2012 Lucas Sherman

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
#include "transferwidget.h"
#include "ui_transferwidget.h"

// Include Dependencies
#include "mainwindow.h"
#include "utilities.h"

// QT4 Includes
#include <QtGui/QMessageBox>

// File scope namespace
namespace 
{
    namespace filescope
    {
        // :TODO: Use a color selector item
        char const* stylesheet = "background-color: %1%";
    }
}

// ----------------------------------------------------------------------------
// Constructor - Initialize widget slots and signals
// ----------------------------------------------------------------------------
TransferWidget::TransferWidget( QWidget *parent ) :
    QWidget( parent ),
    ui( new Ui::TransferWidget )
{
    ui->setupUi(this);

	// Transfer function views
	m_primaryView = new HistogramView( ui->transferPrimary );
	ui->gridLayout_transferPrimary->addWidget( m_primaryView, 0, 0, 1, 1 );
	m_secondaryView = new HistogramView( ui->transferSecondary );
	ui->gridLayout_transferSecondary->addWidget( m_secondaryView, 0, 0, 1, 1 );

	// Default transfer function
	switchDimensions( 1 );

    // Ensure transfer node selection is detected 
    connect(MainWindow::instance, SIGNAL(transferNodeSelected(std::shared_ptr<vox::Node>)),
        this, SLOT(setSelectedNode(std::shared_ptr<vox::Node>)));
}
    
// ----------------------------------------------------------------------------
// Destructor - Delete transfer views
// ----------------------------------------------------------------------------
TransferWidget::~TransferWidget( )
{
    delete ui;
}

// ----------------------------------------------------------------------------
//  Sets the selected node for transfer function editing
// ----------------------------------------------------------------------------
void TransferWidget::setSelectedNode(std::shared_ptr<vox::Node> node)
{
    m_currentNode.reset();
    
    auto material = node->material();

    // Update all of the widget controls
    ui->doubleSpinBox_density->setValue(node->position(0));
    ui->doubleSpinBox_gradient->setValue(node->position(1));
    ui->doubleSpinBox_gradient2->setValue(node->position(2));
    ui->doubleSpinBox_gloss->setValue(material->glossiness());
    ui->doubleSpinBox_opacity->setValue(material->opticalThickness());
    ui->checkBox_visible->setChecked(true);

    m_currentNode = node;
}

// ----------------------------------------------------------------------------
//  Synchronizes the transfer function widget with the active
// ----------------------------------------------------------------------------
void TransferWidget::synchronizeView()
{
    // Update the local scene component handles
    m_transfer    = MainWindow::instance->scene().transfer;
    m_currentNode = m_transfer->nodes().empty() ? 
        nullptr : m_transfer->nodes().front();
}

// ----------------------------------------------------------------------------
//  Applies interactions with the interface to the scene transfer function
//   (Executed on the master renderer thread allowing safe scene changes)
// ----------------------------------------------------------------------------
void TransferWidget::processInteractions()
{
}

// ----------------------------------------------------------------------------
// Returns true if the user allows dimension change
// ----------------------------------------------------------------------------
bool TransferWidget::canSwitchDimensions()
{
	QMessageBox msgBox(this);
	msgBox.setIcon( QMessageBox::Question );
	msgBox.setText( tr("Are you sure you want to switch dimension?\
					Any unsaved changes to the current function will be lost.") );
	msgBox.setWindowTitle( "Switch Dimensions" );

	QPushButton* accept = msgBox.addButton( tr("Yes"), QMessageBox::AcceptRole );
	msgBox.addButton( tr("No"), QMessageBox::RejectRole );
	QPushButton* cancel = msgBox.addButton( tr("Cancel"), QMessageBox::RejectRole );
	msgBox.setDefaultButton( cancel );

	msgBox.exec( );

	if( msgBox.clickedButton( ) != accept ) return false;

	return true;
}

// ----------------------------------------------------------------------------
// Blocks radio button signals 
// ----------------------------------------------------------------------------
void TransferWidget::switchDimensions(int nDims)
{
	// Block radio button signals
	ui->radioButton_1->blockSignals( true );
	ui->radioButton_2->blockSignals( true );
	ui->radioButton_3->blockSignals( true );

	// Switch transfer control context
	switch( nDims ) 
	{
		case 1:

			// Configure widget for 1-Dimensional work 
			ui->radioButton_1->setChecked( true ); 
			ui->groupBox_transferSecondary->hide( );
			ui->horizontalSlider_gradient->hide( );
			ui->doubleSpinBox_gradient->hide( );
			ui->label_gradient->hide( );
			ui->horizontalSlider_gradient2->hide( );
			ui->doubleSpinBox_gradient2->hide( );
			ui->label_gradient2->hide( );
			m_dimensions = 1;
			break;

		case 2:

			// Configure widget for 2-Dimensional work
			ui->radioButton_2->setChecked( true ); 
			ui->groupBox_transferSecondary->hide( );
			ui->horizontalSlider_gradient->show( );
			ui->doubleSpinBox_gradient->show( );
			ui->label_gradient->show( );
			ui->horizontalSlider_gradient2->hide( );
			ui->doubleSpinBox_gradient2->hide( );
			ui->label_gradient2->hide( );
			m_dimensions = 2;
			break;

		case 3:

			// Configure widget for 3-Dimensional work
			ui->radioButton_3->setChecked( true ); 
			ui->groupBox_transferSecondary->show( );
			ui->horizontalSlider_gradient->show( );
			ui->doubleSpinBox_gradient->show( );
			ui->label_gradient->show( );
			ui->horizontalSlider_gradient2->show( );
			ui->doubleSpinBox_gradient2->show( );
			ui->label_gradient2->show( );
			m_dimensions = 3;
			break; 

        default:
            break;
	}

	// Unblock radio button signals
	ui->radioButton_1->blockSignals( false );
	ui->radioButton_2->blockSignals( false );
	ui->radioButton_3->blockSignals( false );
}

// ----------------------------------------------------------------------------
// Single dimension transfer selection
// ----------------------------------------------------------------------------
void TransferWidget::on_radioButton_1_toggled(bool checked)
{
	if (checked && canSwitchDimensions()) switchDimensions(1);
}

// ----------------------------------------------------------------------------
// Single dimension transfer selection
// ----------------------------------------------------------------------------
void TransferWidget::on_radioButton_2_toggled(bool checked)
{
	if (checked && canSwitchDimensions()) switchDimensions(2);
}

// ----------------------------------------------------------------------------
// Single dimension transfer selection
// ----------------------------------------------------------------------------
void TransferWidget::on_radioButton_3_toggled(bool checked)
{	
	if (checked && canSwitchDimensions()) switchDimensions(3);
}

// ----------------------------------------------------------------------------
// Select the first node / previous region
// ----------------------------------------------------------------------------
void TransferWidget::on_pushButton_first_clicked( )
{
    if (m_transfer)
    {
        m_currentNode = m_transfer->nodes().front();
    }
}

// ----------------------------------------------------------------------------
//  Select the next node
// ----------------------------------------------------------------------------
void TransferWidget::on_pushButton_next_clicked( )
{
    if (m_currentNode)
    {
        // Check if this is the last node
        auto nodes = m_transfer->nodes();
        if (m_currentNode != nodes.back())
        {
            // Acquire a handle to the subsequent node in the linked list
            auto iter = std::find(nodes.begin(), nodes.end(), m_currentNode);
            m_currentNode = *(++iter);
        }
    }
}

// ----------------------------------------------------------------------------
//  Select the previous node
// ----------------------------------------------------------------------------
void TransferWidget::on_pushButton_prev_clicked( )
{
    if (m_currentNode)
    {
        // Check if this is the first node
        auto nodes = m_transfer->nodes();
        auto iter  = std::find(nodes.begin(), nodes.end(), m_currentNode);
        if (iter != nodes.begin())
        {
            // Acquire a handle to the previous node in the linked list
            m_currentNode = *(--iter);
        }
    }
}

// ----------------------------------------------------------------------------
//  Select the last node / next region
// ----------------------------------------------------------------------------
void TransferWidget::on_pushButton_last_clicked( )
{
    if (m_transfer)
    {
        m_currentNode = m_transfer->nodes().back();
    }
}

// ----------------------------------------------------------------------------
//  Delete selected node
// ----------------------------------------------------------------------------
void TransferWidget::on_pushButton_delete_clicked( )
{
    if (m_currentNode)
    {
        m_transfer->removeNode(m_currentNode);
    }

    on_pushButton_first_clicked();
}

// ----------------------------------------------------------------------------
//  Select local emission transfer
// ----------------------------------------------------------------------------
void TransferWidget::on_pushButton_emission_clicked( )
{
	QColor color = colorPicker.getColor( Qt::white ); /* currNode->getColor( ); */
	ui->pushButton_emission->setStyleSheet( 
        vox::format( filescope::stylesheet, 
            color.name().toAscii().data() ).c_str( ) );
}

// ----------------------------------------------------------------------------
// Select local specular transfer
// ----------------------------------------------------------------------------
void TransferWidget::on_pushButton_specular_clicked( )
{
	QColor color = colorPicker.getColor( Qt::white ); /* currNode->getColor( ); */
	ui->pushButton_specular->setStyleSheet( 
        vox::format( filescope::stylesheet, 
            color.name().toAscii().data() ).c_str( ) );
}

// ----------------------------------------------------------------------------
// Select local diffuse transfer
// ----------------------------------------------------------------------------
void TransferWidget::on_pushButton_diffuse_clicked( )
{
	QColor color = colorPicker.getColor( Qt::white ); /* currNode->getColor( ); */
	ui->pushButton_diffuse->setStyleSheet( 
        vox::format( filescope::stylesheet, 
            color.name().toAscii().data() ).c_str( ) );
}

// ----------------------------------------------------------------------------
//  Modify the associated double spinbox to reflect slide value change
// ----------------------------------------------------------------------------
void TransferWidget::on_horizontalSlider_density_valueChanged(int value)
{
    Utilities::forceSbToSl(
        ui->doubleSpinBox_density,
        ui->horizontalSlider_density,
        value);

    if (m_currentNode)
    {
        auto value = ui->doubleSpinBox_density->value();
        m_currentNode->setPosition(0, value);
    }
}

// ----------------------------------------------------------------------------
//  Modifies the intensity component of the light's emissions
// ----------------------------------------------------------------------------
void TransferWidget::on_doubleSpinBox_density_valueChanged(double value)
{
    Utilities::forceSlToSb(
        ui->horizontalSlider_density,
        ui->doubleSpinBox_density,
        value);

    if (m_currentNode)
    {
        auto value = ui->doubleSpinBox_density->value();
        m_currentNode->setPosition(0, value);
    }
}

// ----------------------------------------------------------------------------
//  Modify the associated double spinbox to reflect slide value change
// ----------------------------------------------------------------------------
void TransferWidget::on_horizontalSlider_gloss_valueChanged(int value)
{
    Utilities::forceSbToSl(
        ui->doubleSpinBox_gloss,
        ui->horizontalSlider_gloss,
        value);

    if (m_currentNode)
    {
        auto value = ui->doubleSpinBox_gloss->value();
        m_currentNode->material()->setGlossiness(value);
    }
}

// ----------------------------------------------------------------------------
//  Modifies the intensity component of the light's emissions
// ----------------------------------------------------------------------------
void TransferWidget::on_doubleSpinBox_gloss_valueChanged(double value)
{
    Utilities::forceSlToSb(
        ui->horizontalSlider_gloss,
        ui->doubleSpinBox_gloss,
        value);
    
    if (m_currentNode)
    {
        auto value = ui->doubleSpinBox_gloss->value();
        m_currentNode->material()->setGlossiness(value);
    }
}

// ----------------------------------------------------------------------------
//  Modify the associated double spinbox to reflect slide value change
// ----------------------------------------------------------------------------
void TransferWidget::on_horizontalSlider_opacity_valueChanged(int value)
{
    Utilities::forceSbToSl(
        ui->doubleSpinBox_opacity,
        ui->horizontalSlider_opacity,
        value);

    if (m_currentNode)
    {
        //auto value = ui->doubleSpinBox_opacity->value();
        //m_currentNode->material()->setOpticalThickness(value);
    }
}

// ----------------------------------------------------------------------------
//  Modifies the intensity component of the light's emissions
// ----------------------------------------------------------------------------
void TransferWidget::on_doubleSpinBox_opacity_valueChanged(double value)
{
    Utilities::forceSlToSb(
        ui->horizontalSlider_opacity,
        ui->doubleSpinBox_opacity,
        value);

    if (m_currentNode)
    {
        //auto value = ui->doubleSpinBox_opacity->value();
        //m_currentNode->material()->setOpticalThickness(value);
    }
}

