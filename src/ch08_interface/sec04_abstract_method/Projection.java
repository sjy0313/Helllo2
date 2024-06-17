package ch08_interface.sec04_abstract_method;

import ch08_interface.sec05_defualt_method.RemoteControl;

public class Projection implements RemoteControl {

	private int volume;

	@Override
	public void turnOn() {
		// TODO Auto-generated method stub
		// 스텁은 기존 코드를 시뮬레이션하거나 아직 개발되지
		// 않은 코드를 임시로 대치하는 역할을 수행
		System.out.println("Audio를 켭니다.");
	}

	@Override
	public void turnOff() {
		System.out.println("Audio를 끕니다.");
	}

	@Override
	public void setVolume(int volume) {
		if(volume>RemoteControl.MAX_VOLUME) {
			this.volume = RemoteControl.MAX_VOLUME;
		} else if(volume<RemoteControl.MIN_VOLUME) {
			this.volume = RemoteControl.MIN_VOLUME;
		} else {
			this.volume = volume;
		}
		System.out.println("현재 Audio 볼륨: " + volume);

	}

}
